import os
import random
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from pydantic import BaseModel
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped

from rlhfblender.data_collection import RecordedEpisodesContainer
from rlhfblender.data_collection.metrics_processor import process_metrics
from rlhfblender.data_models.agent import BaseAgent


class BenchmarkSummary(BaseModel):
    benchmark_steps: int = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    additional_metrics: dict


class EpisodeRecorder:
    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env | VecEnv,
        n_eval_episodes: int = 2,
        max_steps: int = int(1e4),
        save_path: str = "",
        overwrite: bool = False,
        deterministic: bool = False,
        render: bool = True,
        reset_to_initial_state: bool = True,
        additional_out_attributes: dict[str, Any] | None = None,
        callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ):
        self.agent = agent
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.save_path = save_path
        self.overwrite = overwrite
        self.deterministic = deterministic
        self.render = render
        self.reset_to_initial_state = reset_to_initial_state
        self.additional_out_attributes = additional_out_attributes
        self.callback = callback

        # Initialize buffers and counters
        self.buffers = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "features": [],
            "probs": [],
            "renders": [],
        }
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.n_envs = env.num_envs if isinstance(env, VecEnv) else 1
        self.current_rewards = np.zeros(self.n_envs)
        self.current_lengths = np.zeros(self.n_envs, dtype=int)
        self.episode_counts = np.zeros(self.n_envs, dtype=int)
        self.episode_count_targets = np.array(
            [(self.n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)], dtype=int
        )
        self.is_monitor_wrapped = self.check_monitor_wrapped()
        self.states = None  # For RNN policies
        self.initial_states = None
        self.reset_obs = None

    def check_monitor_wrapped(self):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor

        return (
            is_vecenv_wrapped(self.env, VecMonitor) or self.env.env_is_wrapped(Monitor)[0]
            if isinstance(self.env, VecEnv)
            else False
        )

    def record_episodes(self):
        """Runs the agent in the environment and records episodes."""
        self.initialize_environment()

        while (self.episode_counts < self.episode_count_targets).any() and self.total_steps <= self.max_steps:
            actions = self.agent.act(self.observations)
            additional_outputs = self.collect_agent_outputs(actions)
            self.handle_resets()
            self.update_buffers(actions, additional_outputs)
            self.process_steps()
            self.handle_rendering()
            self.environment_step(actions)

        self.finalize_buffers()
        self.save_episodes()

    def initialize_environment(self):
        seed = random.randint(0, 2000)
        if not isinstance(self.env, VecEnv):
            self.observations, _ = self.env.reset(seed=seed)
            self.rewards = 0
            self.dones = False
            self.infos = {}
            if isinstance(self.env.observation_space, gym.spaces.Dict) and "mission" in self.observations.keys():
                self.infos["mission"] = self.observations["mission"]
                self.infos["seed"] = seed
            self.observations = np.expand_dims(self.observations, axis=0)
            self.rewards = np.expand_dims(self.rewards, axis=0)
            self.dones = np.expand_dims(self.dones, axis=0)
            self.infos = [self.infos]
        else:
            self.env.seed(seed=seed)
            self.observations = self.env.reset()
            self.rewards = np.zeros(self.n_envs)
            self.dones = np.zeros(self.n_envs, dtype=bool)
            self.infos = [{} for _ in range(self.n_envs)]
            if isinstance(self.env.observation_space, gym.spaces.Dict) and "mission" in self.observations[0].keys():
                for i in range(self.n_envs):
                    self.infos[i]["mission"] = self.observations[i]["mission"]
                    self.infos[i]["seed"] = seed
        if self.reset_to_initial_state:
            if isinstance(self.env, VecEnv):
                self.initial_states = tuple(deepcopy(e) for e in self.env.envs)
            else:
                self.initial_states = deepcopy(self.env)
        self.reset_obs = self.observations.copy()

    def collect_agent_outputs(self, actions):
        additional_outputs = self.agent.additional_outputs(
            self.observations,
            actions,
            output_list=["log_probs", "feature_extractor_output", "entropy", "value"],
        )
        return additional_outputs

    def handle_resets(self):
        if not isinstance(self.env, VecEnv) and self.dones:
            seed = random.randint(0, 1000000)
            self.observations, _ = self.env.reset(seed=seed)
            if isinstance(self.env.observation_space, gym.spaces.Dict) and "mission" in self.observations.keys():
                self.infos[0]["mission"] = self.observations["mission"]
                self.infos[0]["seed"] = seed

    def update_buffers(self, actions, additional_outputs):
        self.buffers["obs"].append(np.squeeze(self.observations))
        self.buffers["actions"].append(np.squeeze(actions))
        self.buffers["rewards"].append(np.squeeze(self.rewards))
        self.buffers["dones"].append(np.squeeze(self.dones))
        if "feature_extractor_output" in additional_outputs:
            self.buffers["features"].append(np.squeeze(additional_outputs["feature_extractor_output"]))
        if "log_probs" in additional_outputs:
            self.buffers["probs"].append(additional_outputs["log_probs"])
        self.buffers["infos"].append(self.process_infos(additional_outputs))

    def process_infos(self, additional_outputs):
        infos = []
        for i in range(self.n_envs):
            info = {
                **self.infos[i],
                **{
                    k: (v[i].item() if isinstance(v[i], np.ndarray) else v[i])
                    for k, v in additional_outputs.items()
                    if k not in ["log_probs", "feature_extractor_output"]
                },
            }
            infos.append(info)
        return np.squeeze(infos) if infos else np.squeeze(self.infos)

    def process_steps(self):
        self.current_rewards += np.squeeze(self.rewards)
        self.current_lengths += 1

        for i in range(self.n_envs):
            if self.episode_counts[i] < self.episode_count_targets[i]:
                if self.callback is not None:
                    self.callback(locals(), globals())

                self.total_steps += 1

                if self.dones[i]:
                    self.handle_episode_end(i)
                    self.reset_environment(i)

    def handle_episode_end(self, i):
        if self.is_monitor_wrapped and "episode" in self.infos[i]:
            self.episode_rewards.append(self.infos[i]["episode"]["r"])
            self.episode_lengths.append(self.infos[i]["episode"]["l"])
            self.infos[i]["label"] = "Real Game End"
        else:
            self.episode_rewards.append(self.current_rewards[i])
            self.episode_lengths.append(self.current_lengths[i])

        self.episode_counts[i] += 1
        self.current_rewards[i] = 0
        self.current_lengths[i] = 0
        if self.states is not None:
            self.states[i] *= 0

    def reset_environment(self, i):
        if self.reset_to_initial_state:
            if isinstance(self.env, VecEnv):
                self.env.envs[i] = deepcopy(self.initial_states[i])
            else:
                self.env = deepcopy(self.initial_states)
            self.observations[i] = self.reset_obs[i]

    def handle_rendering(self):
        if self.render:
            render_frame = self.env.render()
            self.buffers["renders"].append(np.squeeze(render_frame))

    def environment_step(self, actions):
        if not isinstance(self.env, VecEnv):
            # squeeze for now, as we are not expecting vectorized/batched environments
            self.observations, self.rewards, terminated, truncated, self.infos = self.env.step(actions)
            self.dones = terminated or truncated
            self.observations = np.expand_dims(self.observations, axis=0)
            self.rewards = np.expand_dims(self.rewards, axis=0)
            self.dones = np.expand_dims(self.dones, axis=0)
            self.infos = [self.infos]
        else:
            self.observations, self.rewards, self.dones, self.infos = self.env.step(actions)

    def finalize_buffers(self):
        if self.render:
            self.buffers["renders"] = np.array(self.buffers["renders"])
        else:
            self.buffers["renders"] = np.zeros(1)

        if len(self.buffers["obs"][0].shape) == 2:
            self.buffers["obs"] = np.expand_dims(self.buffers["obs"], axis=-1)
        else:
            self.buffers["obs"] = np.array(self.buffers["obs"])
        self.buffers["actions"] = np.array(self.buffers["actions"])
        self.buffers["dones"] = np.array(self.buffers["dones"])
        self.buffers["rewards"] = np.array(self.buffers["rewards"])
        self.buffers["features"] = np.array(self.buffers["features"])
        self.buffers["infos"] = np.array(self.buffers["infos"])
        self.buffers["probs"] = np.array(self.buffers["probs"])

    def save_episodes(self):
        if not self.overwrite and os.path.isfile(self.save_path + ".npz"):
            previous_data = np.load(self.save_path + ".npz", allow_pickle=True)
            for key in self.buffers.keys():
                self.buffers[key] = np.concatenate((previous_data[key], self.buffers[key]), axis=0)
            self.episode_rewards = np.concatenate((previous_data["episode_rewards"], self.episode_rewards), axis=0)
            self.episode_lengths = np.concatenate((previous_data["episode_lengths"], self.episode_lengths), axis=0)

        # Recompute metrics
        additional_metrics = process_metrics(
            RecordedEpisodesContainer(
                obs=self.buffers["obs"],
                actions=self.buffers["actions"],
                dones=self.buffers["dones"],
                rewards=self.buffers["rewards"],
                episode_rewards=self.episode_rewards,
                episode_lengths=self.episode_lengths,
                features=self.buffers["features"],
                infos=self.buffers["infos"],
                probs=self.buffers["probs"],
                renders=self.buffers["renders"],
                additional_metrics={},
            )
        )

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(os.path.join(self.save_path + ".npz"), "wb") as f:
            np.savez(
                f,
                obs=self.buffers["obs"],
                rewards=self.buffers["rewards"],
                dones=self.buffers["dones"],
                actions=self.buffers["actions"],
                renders=self.buffers["renders"],
                infos=self.buffers["infos"],
                features=self.buffers["features"],
                probs=self.buffers["probs"],
                episode_rewards=self.episode_rewards,
                episode_lengths=self.episode_lengths,
                additional_metrics=additional_metrics,
            )

    @staticmethod
    def load_episodes(save_path: str) -> RecordedEpisodesContainer:
        """Load episodes from a file."""
        print(f"[INFO] Loading episodes from {save_path}")
        data = np.load(save_path + ".npz", allow_pickle=True)
        print(f"[INFO] Successfully loaded NPZ episodes from {save_path}")

        return RecordedEpisodesContainer(
            obs=data["obs"],
            rewards=data["rewards"],
            dones=data["dones"],
            infos=data["infos"],
            actions=data["actions"],
            renders=data["renders"],
            features=data["features"],
            probs=data["probs"],
            episode_rewards=data["episode_rewards"],
            episode_lengths=data["episode_lengths"],
            additional_metrics=data["additional_metrics"].item(),
        )

    @staticmethod
    def get_aggregated_data(save_path: str) -> BenchmarkSummary:
        """Load episodes from a file and return aggregated data."""
        data = np.load(save_path + ".npz", allow_pickle=True)

        return BenchmarkSummary(
            benchmark_steps=data["rewards"].shape[0],
            episode_lengths=data["episode_lengths"].tolist(),
            episode_rewards=data["episode_rewards"].tolist(),
            additional_metrics=data["additional_metrics"].item(),
        )


def convert_infos(infos: np.ndarray):
    """
    Convert a numpy array of dict objects to a list of dict objects.
    """
    infos = infos.tolist()
    out_infos = []
    for i, convert_info in enumerate(infos):
        out_info = {}
        convert_info = convert_info.item()
        for key, value in convert_info.items():
            if isinstance(value, np.generic):
                out_info[key] = value.item()
        out_info["id"] = i
        out_info["episode step"] = i
        out_infos.append(out_info)
    return out_infos

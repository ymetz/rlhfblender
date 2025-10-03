import os
import pickle
import random
from collections.abc import Callable
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
        persistent_initial_state_path: str | None = None,
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
        self.persistent_initial_state_path = persistent_initial_state_path
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
            "env_states": [],
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
        self.reset_info = None
        self.skip_next_step = False

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

        while (self.episode_counts < self.episode_count_targets).any():
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
            # Try to load persistent initial state
            if self.persistent_initial_state_path and os.path.exists(self.persistent_initial_state_path):
                with open(self.persistent_initial_state_path, 'rb') as f:
                    self.initial_states = pickle.load(f)
                # Load state into environment
                if isinstance(self.env, VecEnv) and hasattr(self.env, 'envs'):
                    for i, env in enumerate(self.env.envs):
                        if i < len(self.initial_states):
                            obs = env.load_state(self.initial_states[i])
                            if obs is not None:
                                self.observations[i] = obs
                else:
                    obs = self.env.load_state(self.initial_states)
                    if obs is not None:
                        self.observations[0] = obs
            else:
                # Generate and save new initial state
                if isinstance(self.env, VecEnv) and hasattr(self.env, 'envs'):
                    self.initial_states = []
                    for i, env in enumerate(self.env.envs):
                        obs_for_env = self.observations[i] if i < len(self.observations) else self.observations[0]
                        self.initial_states.append(env.save_state(obs_for_env))
                else:
                    self.initial_states = self.env.save_state(self.observations[0])
                
                # Save for future checkpoints
                if self.persistent_initial_state_path:
                    os.makedirs(os.path.dirname(self.persistent_initial_state_path), exist_ok=True)
                    with open(self.persistent_initial_state_path, 'wb') as f:
                        pickle.dump(self.initial_states, f)
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
                self.reset_info = {"mission": self.observations["mission"], "seed": seed}

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

        # Save environment state if the environment supports it
        if isinstance(self.env, VecEnv) and hasattr(self.env, 'envs'):
            # For VecEnv, check individual environments for save_state support
            env_states = []
            for i in range(self.n_envs):
                if hasattr(self.env.envs[i], "save_state"):
                    env_states.append(self.env.envs[i].save_state())
                else:
                    env_states.append(None)
            self.buffers["env_states"].append(env_states)
        else:
            # For single environment
            if hasattr(self.env, "save_state"):
                self.buffers["env_states"].append(self.env.save_state())
            else:
                self.buffers["env_states"].append(None)

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

                self.truncate_episode_if_needed(i)

                if self.dones[i]:
                    self.handle_episode_end(i)
                    self.reset_environment(i)
                    if not isinstance(self.env, VecEnv):
                        self.skip_next_step = True

    def truncate_episode_if_needed(self, i: int):
        if self.max_steps is None:
            return

        if self.current_lengths[i] < self.max_steps:
            return

        if self.dones[i]:
            return

        self.dones[i] = True
        self._update_last_done_flag(i)

    def _update_last_done_flag(self, i: int):
        if not self.buffers["dones"]:
            return

        last_done = self.buffers["dones"][-1]

        if np.isscalar(last_done):
            self.buffers["dones"][-1] = True
            return

        if isinstance(last_done, np.ndarray):
            if last_done.ndim == 0:
                self.buffers["dones"][-1] = True
                return
            updated = last_done.copy()
            updated[i] = True
            self.buffers["dones"][-1] = updated
            return

        if isinstance(last_done, list):
            updated = list(last_done)
            updated[i] = True
            self.buffers["dones"][-1] = updated
            return

        self.buffers["dones"][-1] = True

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
            if isinstance(self.env, VecEnv) and hasattr(self.env, 'envs'):
                # Use load_state method to reset to initial state for VecEnv
                if hasattr(self.env.envs[i], 'load_state'):
                    obs = self.env.envs[i].load_state(self.initial_states[i])
                    if obs is not None:
                        self.observations[i] = obs
                    else:
                        self.observations[i] = self.reset_obs[i]
                else:
                    raise RuntimeError(f"Environment {self.env.envs[i]} does not support load_state method.")
            else:
                # Use load_state method to reset to initial state for single environment
                if hasattr(self.env, 'load_state'):
                    obs = self.env.load_state(self.initial_states)
                    if obs is not None:
                        self.observations[0] = obs
                    else:
                        self.observations[0] = self.reset_obs[0]
                else:
                    raise RuntimeError("Environment does not support load_state method.")

    def handle_rendering(self):
        if self.render:
            render_frame = self.env.render()

            # very special case for metaworld (which has a bug in the rendering) - check if the env is a metaworld env
            # and has a camera name starting with "corner"
            # TODO: Remove this special case when the bug is fixed in metaworld/mujoco
            try:
                if (isinstance(self.env, VecEnv) and hasattr(self.env, 'envs') and 
                    hasattr(self.env.envs[0].unwrapped, "camera_name") and 
                    self.env.envs[0].unwrapped.camera_name.startswith("corner")):
                    # Rotate 180 degrees (2 times 90 degrees clockwise)
                    render_frame = np.rot90(render_frame, k=2)
            except AttributeError:
                # If the environment does not have a camera_name attribute, we can skip this check
                pass

            self.buffers["renders"].append(np.squeeze(render_frame))

    def environment_step(self, actions):
        if not isinstance(self.env, VecEnv):
            if self.skip_next_step:
                self.skip_next_step = False
                self.rewards = np.zeros(1)
                self.dones = np.zeros(1, dtype=bool)
                info = {}
                if self.reset_info is not None:
                    if isinstance(self.reset_info, dict):
                        if "mission" in self.reset_info:
                            info["mission"] = self.reset_info["mission"]
                        if "seed" in self.reset_info:
                            info["seed"] = self.reset_info["seed"]
                    self.reset_info = None
                self.infos = [info]
                return
            # squeeze for now, as we are not expecting vectorized/batched environments
            self.observations, self.rewards, terminated, truncated, self.infos = self.env.step(actions)
            self.dones = terminated or truncated
            self.observations = np.expand_dims(self.observations, axis=0)
            self.rewards = np.expand_dims(self.rewards, axis=0)
            self.dones = np.expand_dims(self.dones, axis=0)
            self.infos = [self.infos]
            if self.reset_info is not None:
                self.infos[0]["mission"] = self.reset_info["mission"]
                self.infos[0]["seed"] = self.reset_info["seed"]
                self.reset_info = None
        else:
            self.observations, self.rewards, self.dones, self.infos = self.env.step(actions)

    def finalize_buffers(self):
        print(len(self.buffers["obs"]), len(self.buffers["actions"]), len(self.buffers["rewards"]))
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

        # Handle env_states - keep as object array to handle None values and varying structures
        if len(self.buffers["env_states"]) > 0:
            self.buffers["env_states"] = np.array(self.buffers["env_states"], dtype=object)
        else:
            self.buffers["env_states"] = np.array([], dtype=object)

    def save_episodes(self):
        if not self.overwrite and os.path.isfile(self.save_path + ".npz"):
            previous_data = np.load(self.save_path + ".npz", allow_pickle=True)
            for key in self.buffers.keys():
                if key in previous_data:
                    self.buffers[key] = np.concatenate((previous_data[key], self.buffers[key]), axis=0)
                # If env_states wasn't in previous data, just keep the new data
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
                env_states=self.buffers["env_states"],
                additional_metrics={},
            )
        )

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print("[INFO] Saving episodes to", self.save_path + ".npz")
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
                env_states=self.buffers["env_states"],
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
            env_states=data.get("env_states", np.array([], dtype=object)),  # Handle backward compatibility
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

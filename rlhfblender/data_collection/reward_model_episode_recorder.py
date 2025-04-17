import argparse
import asyncio
import os
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from databases import Database
from multi_type_feedback.networks import SingleCnnNetwork, SingleNetwork
from pydantic import BaseModel
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped

from rlhfblender.data_collection import RecordedEpisodesContainer
from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.data_collection.metrics_processor import process_metrics
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.agent import BaseAgent, RandomAgent, TrainedAgent
from rlhfblender.data_models.global_models import Environment, Experiment

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))


class BenchmarkSummary(BaseModel):
    benchmark_steps: int = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    additional_metrics: dict


@dataclass
class EnhancedRecordedEpisodesContainer:
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    actions: np.ndarray
    infos: np.ndarray
    renders: np.ndarray
    features: np.ndarray
    probs: np.ndarray
    predicted_rewards: np.ndarray
    uncertainties: np.ndarray
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    additional_metrics: dict[str, Any]


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
        reward_model_path: str = None,
        random_reward_model: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        self.reward_model_path = reward_model_path
        self.random_reward_model = random_reward_model
        self.device = device

        # Setup reward model if provided
        self.reward_model = None
        if self.reward_model_path:
            self.setup_reward_model()

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
            "predicted_rewards": [],
            "uncertainties": [],
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

    def setup_reward_model(self):
        """Initialize and load the reward model."""
        # Determine if we need CNN network based on environment observation space
        obs_space = self.env.observation_space

        # Check if we need CNN by checking observation space dimensions
        if len(obs_space.shape) > 1 and obs_space.shape[0] > 1 and obs_space.shape[1] > 1:
            # This is likely image observation space (e.g., Atari games)
            reward_model_cls = SingleCnnNetwork
        else:
            # This is likely a vector observation space (e.g., MuJoCo tasks)
            reward_model_cls = SingleNetwork

        self.reward_model = reward_model_cls.load_from_checkpoint(self.reward_model_path, map_location=self.device)
        self.reward_model.eval()  # Set to evaluation mode
        print(f"Loaded reward model from {self.reward_model_path}")

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

            # Get reward model predictions if available
            if self.reward_model:
                predicted_reward, uncertainty = self.get_reward_model_predictions(self.observations, actions)
            else:
                predicted_reward, uncertainty = np.zeros(self.n_envs), np.zeros(self.n_envs)

            self.handle_resets()
            self.update_buffers(actions, additional_outputs, predicted_reward, uncertainty)
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
                self.reset_info = {"mission": self.observations["mission"], "seed": seed}

    def update_buffers(self, actions, additional_outputs, predicted_reward, uncertainty):
        self.buffers["obs"].append(np.squeeze(self.observations))
        self.buffers["actions"].append(np.squeeze(actions))
        self.buffers["rewards"].append(np.squeeze(self.rewards))
        self.buffers["dones"].append(np.squeeze(self.dones))
        self.buffers["predicted_rewards"].append(np.squeeze(predicted_reward))
        self.buffers["uncertainties"].append(np.squeeze(uncertainty))

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
            if self.reset_info is not None:
                self.infos[0]["mission"] = self.reset_info["mission"]
                self.infos[0]["seed"] = self.reset_info["seed"]
                self.reset_info = None
        else:
            self.observations, self.rewards, self.dones, self.infos = self.env.step(actions)

    def get_reward_model_predictions(self, observations, actions):
        """
        Get reward model predictions and uncertainty for a given state-action pair.

        Returns:
            tuple: (predicted_rewards, uncertainties)
        """
        with torch.no_grad():
            # Convert to tensors and move to device
            obs_tensor = torch.as_tensor(observations, device=self.device, dtype=torch.float)

            # Handle actions based on their type (discrete vs continuous)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action_tensor = self._one_hot_encode_actions(actions, self.env.action_space.n)
            else:
                action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float)

            # Reshape for reward model if needed based on model's expected input format
            batch_size = obs_tensor.shape[0]

            # Add sequence dimension needed by reward model (batch_size, seq_len, ...)
            if len(obs_tensor.shape) == 2:  # (batch_size, obs_dim)
                obs_tensor = obs_tensor.unsqueeze(1)  # Add sequence dim
            elif len(obs_tensor.shape) > 2 and obs_tensor.shape[1] != 1:  # Reshape if needed
                # For CNN inputs, likely (batch_size, channels, height, width)
                # Rewire to (batch_size, 1, channels, height, width)
                obs_tensor = obs_tensor.unsqueeze(1)

            # Add sequence dimension to actions if needed
            if len(action_tensor.shape) == 2:  # (batch_size, action_dim)
                action_tensor = action_tensor.unsqueeze(1)

            # Get predictions
            if self.reward_model.ensemble_count > 1:
                # For ensemble models, expand inputs to ensemble_count
                obs_tensor = obs_tensor.expand(self.reward_model.ensemble_count, *obs_tensor.shape[1:])
                action_tensor = action_tensor.expand(self.reward_model.ensemble_count, *action_tensor.shape[1:])

                predictions = self.reward_model(obs_tensor, action_tensor)

                # Reshape predictions to get ensemble rewards per sample
                # Shape: (ensemble_count, batch_size, 1)
                predictions = predictions.view(self.reward_model.ensemble_count, batch_size, -1)

                # Calculate mean and std across ensemble models
                mean_rewards = torch.mean(predictions, dim=0).squeeze(-1)
                uncertainties = torch.std(predictions, dim=0).squeeze(-1)
            else:
                # For single models
                predictions = self.reward_model(obs_tensor, action_tensor)
                mean_rewards = predictions.squeeze(-1)
                # No real uncertainty for non-ensemble models, return zeros
                uncertainties = torch.zeros_like(mean_rewards)

            return mean_rewards.cpu().numpy(), uncertainties.cpu().numpy()

    def _one_hot_encode_actions(self, actions, n_actions):
        """
        Convert discrete actions to one-hot encoding.

        Args:
            actions: Numpy array of discrete actions
            n_actions: Number of possible actions

        Returns:
            torch.Tensor: One-hot encoded actions
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long()
        else:
            actions = torch.tensor(actions, dtype=torch.long)

        # Handle single action case
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)

        one_hot = torch.zeros(actions.shape[0], n_actions, device=self.device)
        one_hot.scatter_(1, actions.unsqueeze(1), 1)
        return one_hot

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
        self.buffers["predicted_rewards"] = np.array(self.buffers["predicted_rewards"])
        self.buffers["uncertainties"] = np.array(self.buffers["uncertainties"])
        self.buffers["features"] = np.array(self.buffers["features"])
        self.buffers["infos"] = np.array(self.buffers["infos"])
        self.buffers["probs"] = np.array(self.buffers["probs"])

    def save_episodes(self):
        if not self.overwrite and os.path.isfile(self.save_path + ".npz"):
            previous_data = np.load(self.save_path + ".npz", allow_pickle=True)
            for key in self.buffers.keys():
                if key in previous_data:
                    self.buffers[key] = np.concatenate((previous_data[key], self.buffers[key]), axis=0)
            self.episode_rewards = np.concatenate((previous_data["episode_rewards"], self.episode_rewards), axis=0)
            self.episode_lengths = np.concatenate((previous_data["episode_lengths"], self.episode_lengths), axis=0)

        # Recompute metrics
        additional_metrics = process_metrics(
            EnhancedRecordedEpisodesContainer(
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
                predicted_rewards=self.buffers["predicted_rewards"],
                uncertainties=self.buffers["uncertainties"],
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
                predicted_rewards=self.buffers["predicted_rewards"],
                uncertainties=self.buffers["uncertainties"],
                episode_rewards=self.episode_rewards,
                episode_lengths=self.episode_lengths,
                additional_metrics=additional_metrics,
            )

    @staticmethod
    def load_episodes(save_path: str) -> EnhancedRecordedEpisodesContainer:
        """Load episodes from a file."""
        print(f"[INFO] Loading episodes from {save_path}")
        data = np.load(save_path + ".npz", allow_pickle=True)
        print(f"[INFO] Successfully loaded NPZ episodes from {save_path}")

        # Handle case where file was saved without predicted rewards/uncertainties
        predicted_rewards = data["predicted_rewards"] if "predicted_rewards" in data else np.zeros_like(data["rewards"])
        uncertainties = data["uncertainties"] if "uncertainties" in data else np.zeros_like(data["rewards"])

        return EnhancedRecordedEpisodesContainer(
            obs=data["obs"],
            rewards=data["rewards"],
            dones=data["dones"],
            infos=data["infos"],
            actions=data["actions"],
            renders=data["renders"],
            features=data["features"],
            probs=data["probs"],
            predicted_rewards=predicted_rewards,
            uncertainties=uncertainties,
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

    @staticmethod
    def convert_to_feedback_dataset(
        episodes: EnhancedRecordedEpisodesContainer,
        feedback_type: str = "evaluative",
        human_preds: Optional[list[float]] = None,
    ) -> dict:
        """
        Convert recorded episodes to a format suitable for FeedbackDataset.

        Args:
            episodes: Recorded episodes container
            feedback_type: Type of feedback ('evaluative', 'comparative', etc.)
            human_preds: Human-provided ratings or preferences (optional)

        Returns:
            dict: Feedback data structure compatible with FeedbackDataset
        """
        if feedback_type not in ["evaluative", "comparative", "demonstrative", "corrective", "descriptive"]:
            raise ValueError(f"Unsupported feedback type: {feedback_type}")

        # Extract episode boundaries
        episode_end_indices = []
        current_idx = 0
        for ep_len in episodes.episode_lengths:
            current_idx += ep_len
            episode_end_indices.append(current_idx - 1)  # -1 because indices are 0-based

        # Create segments based on episode boundaries
        segments = []
        start_idx = 0

        for end_idx in episode_end_indices:
            # Extract the current episode
            episode_obs = episodes.obs[start_idx : end_idx + 1]
            episode_actions = episodes.actions[start_idx : end_idx + 1]
            episode_rewards = episodes.rewards[start_idx : end_idx + 1]
            episode_dones = episodes.dones[start_idx : end_idx + 1]

            # Create segment tuples (obs, action, reward, done)
            episode_segment = []
            for i in range(len(episode_obs)):
                episode_segment.append(
                    (
                        np.expand_dims(episode_obs[i], axis=0),  # Add batch dimension
                        episode_actions[i],
                        episode_rewards[i],
                        episode_dones[i],
                    )
                )

            segments.append(episode_segment)
            start_idx = end_idx + 1

        # Construct feedback data based on feedback type
        feedback_data = {}

        if feedback_type == "evaluative":
            feedback_data["segments"] = segments
            # Use human_preds if provided, otherwise use model predicted rewards
            if human_preds is not None:
                feedback_data["ratings"] = human_preds
            else:
                # Compute average predicted reward for each segment
                avg_predicted_rewards = []
                start_idx = 0
                for ep_len in episodes.episode_lengths:
                    segment_rewards = episodes.predicted_rewards[start_idx : start_idx + ep_len]
                    avg_predicted_rewards.append(float(np.mean(segment_rewards)))
                    start_idx += ep_len
                feedback_data["ratings"] = avg_predicted_rewards

        elif feedback_type == "comparative":
            # For comparative feedback, we need pairs of segments
            # Here we'll create all possible pairs of segments
            if len(segments) < 2:
                raise ValueError("Need at least 2 segments for comparative feedback")

            feedback_data["segments"] = segments

            # Create all possible pairs for preferences
            preferences = []
            opt_gaps = []

            # Compute "optimality gap" based on predicted rewards
            for i, segment in enumerate(segments):
                segment_rewards = [s[2] for s in segment]  # Extract rewards
                total_reward = sum(segment_rewards)
                opt_gaps.append(-total_reward)  # Negative because lower is better

            # Create preference pairs
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    # Determine preference based on total reward
                    pref = 0 if opt_gaps[i] < opt_gaps[j] else 1
                    preferences.append((i, j, pref))

            feedback_data["preferences"] = preferences
            feedback_data["opt_gaps"] = opt_gaps

        elif feedback_type == "demonstrative":
            # For demonstrative feedback, we use the segments as demonstrations
            feedback_data["demos"] = segments

        elif feedback_type == "descriptive":
            # For descriptive feedback, create cluster representatives
            description = []
            for i, segment in enumerate(segments):
                # Take the first state-action pair from each segment as representative
                first_obs = segment[0][0]
                first_action = segment[0][1]

                # Compute average reward for the segment
                avg_reward = np.mean([s[2] for s in segment])

                description.append((first_obs, first_action, float(avg_reward)))

            feedback_data["description"] = description

            # Also create description preferences
            if len(description) >= 2:
                description_preference = []
                for i in range(len(description)):
                    for j in range(i + 1, len(description)):
                        # Preference based on reward
                        pref = 0 if description[i][2] > description[j][2] else 1
                        description_preference.append((i, j, pref))

                feedback_data["description_preference"] = description_preference

        return feedback_data


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


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="CartPole-v1")
    argparser.add_argument("--exp", type=str, default=None, help="(Optional)  - Load experiment from db")
    argparser.add_argument("--n_eval_episodes", type=int, default=2)
    argparser.add_argument("--max_steps", type=int, default=1000)
    argparser.add_argument("--save_path", type=str, default="data/reward_episodes")
    argparser.add_argument("--overwrite", action="store_true")
    argparser.add_argument("--deterministic", action="store_true")
    argparser.add_argument("--render", action="store_true")
    argparser.add_argument("--reset_to_initial_state", action="store_true")
    argparser.add_argument("--policy_path", type=str, default=None)
    argparser.add_argument("--reward_model_path", type=str, default=None)
    argparser.add_argument("--random_reward_model", action="store_true")
    argparser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = argparser.parse_args()

    if args.exp:
        # Load experiment from database
        async def get_exp_and_env():
            db_experiment = await db_handler.get_single_entry(
                database,
                Experiment,
                key=args.exp,
                key_column="exp_name",
            )
            db_env = await db_handler.get_single_entry(
                database,
                Environment,
                key=db_experiment.env_id,
                key_column="env_name",
            )
            env = get_environment(
                db_env.registration_id,
                n_envs=1,
                environment_config=db_experiment.environment_config,
                additional_packages=db_env.additional_gym_packages,
                gym_entry_point=db_env.gym_entry_point,
            )
            return db_experiment, env

        db_env, env = asyncio.run(get_exp_and_env())

    else:
        env = get_environment(args.env)

    if args.random_reward_model:
        args.reward_model_path = None
        agent = RandomAgent(env.observation_space, env.action_space, env=env)
    else:
        if args.policy_path:
            agent = TrainedAgent.load(args.policy_path, env=env, device=args.device)
        else:
            raise ValueError("Policy path is required for loading an agent.")

    # Initialize with a reward model path
    recorder = EpisodeRecorder(
        agent=agent,
        env=env,
        n_eval_episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        save_path=args.save_path,
        overwrite=args.overwrite,
        deterministic=args.deterministic,
        render=args.render,
        reset_to_initial_state=args.reset_to_initial_state,
        additional_out_attributes={"additional_info": "example"},
        callback=None,
        reward_model_path=args.reward_model_path,
        device=args.device,
    )

    # Record episodes - now includes reward model predictions
    recorder.record_episodes()

    # Load recorded episodes
    episodes = EpisodeRecorder.load_episodes(recorder.save_path)

    # Convert to feedback dataset format
    # If you have human feedback:
    """feedback_data = EpisodeRecorder.convert_to_feedback_dataset(
        episodes, 
        feedback_type="evaluative",
        human_preds=[8.5, 7.2, 9.0]  # Human ratings
    )"""

    # Or using model predictions:
    """feedback_data = EpisodeRecorder.convert_to_feedback_dataset(
        episodes, 
        feedback_type="evaluative"
    )"""

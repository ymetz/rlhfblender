import pickle
from typing import Any, List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize


def one_hot_vector(k, max_val):
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec


class FeedbackOracle:

    def __init__(
        self,
        expert_models: List[Tuple[Union[PPO, SAC], VecNormalize]],
        environment,
        reference_data_path: str,  # Path to pre-computed reference data
        segment_len: int = 50,
        gamma: float = 0.99,
        noise_level: float = 0.0,
        n_clusters: int = 100,
    ):
        """
        Generate on-the-fly oracle feedback for queries and different feedback types
        """
        self.expert_models = expert_models
        self.environment = environment
        self.segment_len = segment_len
        self.gamma = gamma
        self.noise_level = noise_level
        self.n_clusters = n_clusters
        self.action_one_hot = isinstance(self.environment.action_space, gym.spaces.Discrete)
        if self.action_one_hot:
            self.one_hot_dim = self.environment.action_space.n

        # Load and process reference data for calibration
        self.initialize_calibration(reference_data_path)

    def initialize_calibration(self, reference_data_path: str):
        """Initialize calibration data from pre-computed reference trajectories."""
        with open(reference_data_path, "rb") as f:
            reference_data = pickle.load(f)

        # Store reference optimality gaps for evaluative feedback calibration
        self.reference_opt_gaps = reference_data["opt_gaps"]
        max_rating = 10
        self.ratings_bins = np.linspace(min(self.reference_opt_gaps), max(self.reference_opt_gaps), max_rating + 1)

        # Process state-action pairs for clustering (descriptive feedback)
        states_actions = []
        for segment in reference_data["segments"]:
            states_actions.extend([np.concatenate((step[0].squeeze(0), step[1])) for step in segment])
        states_actions = np.array(states_actions)

        # Fit clustering model
        batch_size = min(1000, len(states_actions) // 100)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size, random_state=42)
        self.kmeans.fit(states_actions)

        # Store cluster representatives and their average returns
        self.cluster_representatives = []
        self.cluster_rewards = []

        # Calculate average rewards for each cluster
        cluster_assignments = self.kmeans.predict(states_actions)
        rewards = []
        for segment in reference_data["segments"]:
            rewards.extend([step[2] for step in segment])
        rewards = np.array(rewards)

        for i in range(self.n_clusters):
            cluster_mask = cluster_assignments == i
            if np.any(cluster_mask):
                center = self.kmeans.cluster_centers_[i]
                avg_reward = np.mean(rewards[cluster_mask])
                self.cluster_representatives.append(center)
                self.cluster_rewards.append(avg_reward)

        self.cluster_representatives = np.array(self.cluster_representatives)
        self.cluster_rewards = np.array(self.cluster_rewards)

    def get_feedback(
        self,
        trajectory_data: Union[
            List[Tuple[np.ndarray, np.ndarray, float, bool]],  # single trajectory
            Tuple[
                List[Tuple[np.ndarray, np.ndarray, float, bool]],
                List[Tuple[np.ndarray, np.ndarray, float, bool]],
            ],
        ],
        initial_state: np.ndarray,
        feedback_type: str,
    ) -> Any:
        """
        Get feedback of specified type for either a single trajectory or a pair of trajectories.

        Args:
            trajectory_data: Either a single trajectory or a tuple of two trajectories
            initial_state: Initial state of the trajectory for demonstrative/corrective feedback
            feedback_type: Type of feedback to provide

        Returns:
            Feedback of the specified type
        """
        if feedback_type in [
            "evaluative",
            "demonstrative",
            "corrective",
            "descriptive",
            "supervised",
        ]:
            if not isinstance(trajectory_data, list):
                raise ValueError(f"{feedback_type} feedback requires a single trajectory")

            if feedback_type == "evaluative":
                return self.get_evaluative_feedback(trajectory_data)
            elif feedback_type == "demonstrative":
                return self.get_demonstrative_feedback(initial_state)
            elif feedback_type == "corrective":
                return self.get_corrective_feedback(trajectory_data, initial_state)
            elif feedback_type == "descriptive":
                return self.get_descriptive_feedback(trajectory_data)
            elif feedback_type == "supervised":
                return self.get_supervised_feedback(trajectory_data)

        elif feedback_type in ["comparative", "descriptive_preference"]:
            if not isinstance(trajectory_data, tuple) or len(trajectory_data) != 2:
                raise ValueError(f"{feedback_type} feedback requires a pair of trajectories")

            trajectory1, trajectory2 = trajectory_data

            if feedback_type == "comparative":
                return self.get_comparative_feedback(trajectory1, trajectory2)
            elif feedback_type == "descriptive_preference":
                return self.get_descriptive_preference_feedback(trajectory1, trajectory2)

        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

    def get_supervised_feedback(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], float]]:
        """
        Ground-Truth Fully Supervised Rewards
        """
        return [
            (
                (
                    torch.as_tensor(p[0]).unsqueeze(0).float(),
                    torch.as_tensor(p[1]).unsqueeze(0).float(),
                    torch.ones(1).unsqueeze(-1),
                ),
                torch.as_tensor(p[2]).float(),
            )
            for p in trajectory
        ]

    def get_evaluative_feedback(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        obs = torch.vstack([torch.as_tensor(p[0]).float() for p in trajectory])
        actions = torch.vstack([torch.as_tensor(p[1]).float() for p in trajectory])

        # Pad if necessary
        mask = torch.ones(len(trajectory)).unsqueeze(-1)
        if len(trajectory) < self.segment_len:
            pad_size = self.segment_len - len(trajectory)
            obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
            actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_size, 1)], dim=0)

        # Calculate rating
        opt_gap = -self._compute_discounted_return(trajectory)

        if self.noise_level > 0:
            opt_gap += np.random.normal(0, self.noise_level * np.std(self.reference_opt_gaps))

        bin_index = np.digitize(opt_gap, self.ratings_bins) - 1
        rating = 10 - bin_index
        rating = max(0, min(10, rating))

        # for simplicity, we normalize the rating form 0-0.9

        return (obs, actions, mask), rating / 10

    def get_comparative_feedback(
        self,
        trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ):
        return1 = self._compute_discounted_return(trajectory1)
        return2 = self._compute_discounted_return(trajectory2)

        # trajectory 1
        trajectory1_obs = torch.vstack([torch.as_tensor(p[0]).float() for p in trajectory1])
        trajectory1_actions = torch.vstack([torch.as_tensor(p[1]).float() for p in trajectory1])

        # trajectory 2
        trajectory2_obs = torch.vstack([torch.as_tensor(p[0]).float() for p in trajectory2])
        trajectory2_actions = torch.vstack([torch.as_tensor(p[1]).float() for p in trajectory2])

        if self.noise_level > 0:
            return1 += np.random.normal(0, self.noise_level * abs(return1))
            return2 += np.random.normal(0, self.noise_level * abs(return2))

        mask1 = torch.ones(len(trajectory1)).unsqueeze(-1)
        if len(trajectory1) < self.segment_len:
            pad_size = self.segment_len - len(trajectory1)
            trajectory1_obs = torch.cat(
                [trajectory1_obs, torch.zeros(pad_size, *trajectory1_obs.shape[1:])],
                dim=0,
            )
            trajectory1_actions = torch.cat(
                [
                    trajectory1_actions,
                    torch.zeros(pad_size, *trajectory1_actions.shape[1:]),
                ],
                dim=0,
            )
            mask1 = torch.cat([mask1, torch.zeros(pad_size, 1)], dim=0)

        mask2 = torch.ones(len(trajectory2)).unsqueeze(-1)
        if len(trajectory2) < self.segment_len:
            pad_size = self.segment_len - len(trajectory2)
            trajectory2_obs = torch.cat(
                [trajectory2_obs, torch.zeros(pad_size, *trajectory2_obs.shape[1:])],
                dim=0,
            )
            trajectory2_actions = torch.cat(
                [
                    trajectory2_actions,
                    torch.zeros(pad_size, *trajectory2_actions.shape[1:]),
                ],
                dim=0,
            )
            mask2 = torch.cat([mask2, torch.zeros(pad_size, 1)], dim=0)

        total_return = abs(return1) + abs(return2)
        if total_return == 0:
            return (
                (trajectory1_obs, trajectory1_actions, mask1),
                (trajectory2_obs, trajectory2_actions, mask2),
            ), 0

        diff = abs(return1 - return2) / total_return
        if return1 > return2:
            return (
                (trajectory2_obs, trajectory2_actions, mask2),
                (trajectory1_obs, trajectory1_actions, mask1),
            ), 1
        else:
            return (
                (trajectory1_obs, trajectory1_actions, mask1),
                (trajectory2_obs, trajectory2_actions, mask2),
            ), 1

    def get_demonstrative_feedback(
        self, initial_state
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], int]:
        """Return demonstration and random trajectory pair with preference label 1."""
        demo = self._get_best_demonstration(initial_state)
        random_trajectory = self.get_random_trajectory()

        # Convert demo to tensor format
        obs_demo = torch.vstack([torch.as_tensor(p[0]).float() for p in demo])
        actions_demo = torch.vstack([torch.as_tensor(p[1]).float() for p in demo])

        # Convert random trajectory to tensor format
        obs_rand = torch.vstack([torch.as_tensor(p[0]).float() for p in random_trajectory])
        actions_rand = torch.vstack([torch.as_tensor(p[1]).float() for p in random_trajectory])

        # Pad both trajectories if necessary
        mask_demo = torch.ones(len(demo)).unsqueeze(-1)
        if len(demo) < self.segment_len:
            pad_size = self.segment_len - len(demo)
            obs_demo = torch.cat([obs_demo, torch.zeros(pad_size, *obs_demo.shape[1:])], dim=0)
            actions_demo = torch.cat([actions_demo, torch.zeros(pad_size, *actions_demo.shape[1:])], dim=0)
            mask_demo = torch.cat([mask_demo, torch.zeros(pad_size, 1)], dim=0)

        mask_rand = torch.ones(len(random_trajectory)).unsqueeze(-1)
        if len(random_trajectory) < self.segment_len:
            pad_size = self.segment_len - len(random_trajectory)
            obs_rand = torch.cat([obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0)
            actions_rand = torch.cat([actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])], dim=0)
            mask_rand = torch.cat([mask_rand, torch.zeros(pad_size, 1)], dim=0)

        return (
            (obs_rand, actions_rand, mask_rand),
            (obs_demo, actions_demo, mask_demo),
        ), 1

    def get_corrective_feedback(
        self, trajectory, initial_state
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], int]:
        """Return original and corrected trajectory pair with preference label 1."""
        trajectory_return = self._compute_discounted_return(trajectory)
        expert_demo = self._get_best_demonstration(initial_state)
        expert_return = self._compute_discounted_return(expert_demo)

        if self.noise_level > 0:
            expert_return += np.random.normal(0, self.noise_level * abs(expert_return))
            trajectory_return += np.random.normal(0, self.noise_level * abs(trajectory_return))

        # Convert trajectories to tensor format
        obs_orig = torch.vstack([torch.as_tensor(p[0]).float() for p in trajectory])
        actions_orig = torch.vstack([torch.as_tensor(p[1]).float() for p in trajectory])

        obs_expert = torch.vstack([torch.as_tensor(p[0]).float() for p in expert_demo])
        actions_expert = torch.vstack([torch.as_tensor(p[1]).float() for p in expert_demo])

        # Pad if necessary
        mask_traj = torch.ones(len(trajectory)).unsqueeze(-1)
        if len(trajectory) < self.segment_len:
            pad_size = self.segment_len - len(trajectory)
            obs_orig = torch.cat([obs_orig, torch.zeros(pad_size, *obs_orig.shape[1:])], dim=0)
            actions_orig = torch.cat([actions_orig, torch.zeros(pad_size, *actions_orig.shape[1:])], dim=0)
            mask_traj = torch.cat([mask_traj, torch.zeros(pad_size, 1)], dim=0)

        mask_demo = torch.ones(len(expert_demo)).unsqueeze(-1)
        if len(expert_demo) < self.segment_len:
            pad_size = self.segment_len - len(expert_demo)
            obs_expert = torch.cat([obs_expert, torch.zeros(pad_size, *obs_expert.shape[1:])], dim=0)
            actions_expert = torch.cat(
                [actions_expert, torch.zeros(pad_size, *actions_expert.shape[1:])],
                dim=0,
            )
            mask_demo = torch.cat([mask_demo, torch.zeros(pad_size, 1)], dim=0)

        if expert_return > trajectory_return:
            return (
                (obs_orig, actions_orig, mask_traj),
                (obs_expert, actions_expert, mask_demo),
            ), 1
        return (
            (obs_expert, actions_expert, mask_demo),
            (obs_orig, actions_orig, mask_traj),
        ), 1

    def get_descriptive_feedback(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the most similar cluster representative and its average reward."""
        # Compute average state-action for the trajectory
        states_actions = np.array([np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory])
        avg_state_action = np.mean(states_actions, axis=0)

        # Find closest cluster
        distances = cdist([avg_state_action], self.cluster_representatives)
        closest_cluster = np.argmin(distances)

        # Get cluster representative
        representative = self.cluster_representatives[closest_cluster]
        reward = self.cluster_rewards[closest_cluster]

        # Add noise to reward if specified
        if self.noise_level > 0:
            reward += np.random.normal(0, self.noise_level * np.std(self.cluster_rewards))

        # Split into state and action components
        obs_dim = trajectory[0][0].squeeze(0).shape[0]
        return (
            torch.as_tensor(representative[:obs_dim]).unsqueeze(0).float(),  # state
            torch.as_tensor(representative[obs_dim:]).unsqueeze(0).float(),  # action
            torch.ones(1).unsqueeze(-1),  # mask (for compatability)
        ), reward

    def get_descriptive_preference_feedback(
        self,
        trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ) -> Tuple[int, int, int]:
        """Compare two trajectories based on their closest cluster representatives."""
        # Get cluster information for both trajectories
        states_actions1 = np.array([np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory1])
        states_actions2 = np.array([np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory2])

        avg_state_action1 = np.mean(states_actions1, axis=0)
        avg_state_action2 = np.mean(states_actions2, axis=0)

        # Find closest clusters
        distances1 = cdist([avg_state_action1], self.cluster_representatives)
        distances2 = cdist([avg_state_action2], self.cluster_representatives)

        cluster1 = np.argmin(distances1)
        cluster2 = np.argmin(distances2)

        representative1 = self.cluster_representatives[cluster1]
        representative2 = self.cluster_representatives[cluster2]

        reward1 = self.cluster_rewards[cluster1]
        reward2 = self.cluster_rewards[cluster2]

        # Add noise if specified
        if self.noise_level > 0:
            noise_scale = self.noise_level * np.std(self.cluster_rewards)
            reward1 += np.random.normal(0, noise_scale)
            reward2 += np.random.normal(0, noise_scale)

        # Compare cluster rewards
        reward_diff = reward1 - reward2
        total_reward = abs(reward1) + abs(reward2)
        diff = abs(reward_diff) / total_reward

        obs_dim = trajectory1[0][0].squeeze(0).shape[0]
        if reward1 > reward2:
            return (
                (
                    torch.as_tensor(representative2[:obs_dim]).unsqueeze(0).float(),  # state
                    torch.as_tensor(representative2[obs_dim:]).unsqueeze(0).float(),  # action,
                    torch.ones(1).unsqueeze(-1),  # mask (for compatability)
                ),
                (
                    torch.as_tensor(representative1[:obs_dim]).unsqueeze(0).float(),  # state
                    torch.as_tensor(representative1[obs_dim:]).unsqueeze(0).float(),  # action
                    torch.ones(1).unsqueeze(-1),  # mask (for compatability)
                ),
            ), 1
        else:
            return (
                (
                    torch.as_tensor(representative1[:obs_dim]).unsqueeze(0).float(),  # state
                    torch.as_tensor(representative1[obs_dim:]).unsqueeze(0).float(),  # action
                    torch.ones(1).unsqueeze(-1),  # mask (for compatability)
                ),
                (
                    torch.as_tensor(representative2[:obs_dim]).unsqueeze(0).float(),  # state
                    torch.as_tensor(representative2[obs_dim:]).unsqueeze(0).float(),  # action
                    torch.ones(1).unsqueeze(-1),  # mask (for compatability)
                ),
            ), 1

    def get_random_trajectory(self):
        """Existing implementation..."""
        self.environment.reset()
        trajectory = []
        done = False

        while not done and len(trajectory) < self.segment_len:
            action = self.environment.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            if self.action_one_hot:
                action = one_hot_vector(action, self.one_hot_dim)
            trajectory.append((np.expand_dims(next_obs, axis=0), action, reward, done))

        return trajectory

    def _compute_discounted_return(self, trajectory: List[Tuple]) -> float:
        """Helper method to compute discounted return of a trajectory."""
        rewards = [step[2] for step in trajectory]
        return sum(reward * (self.gamma**i) for i, reward in enumerate(rewards))

    def _get_best_demonstration(self, initial_state):
        """Helper method to get the best demonstration from expert models."""
        best_demo = None
        best_return = float("-inf")

        for exp_model_index, (expert_model, exp_norm_env) in enumerate(self.expert_models):
            self.environment.reset()
            obs = self.environment.load_state(initial_state)
            demo = []

            for _ in range(self.segment_len):
                action, _ = expert_model.predict(
                    exp_norm_env.normalize_obs(obs) if exp_norm_env else obs,
                    deterministic=True,
                )
                next_obs, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                if self.action_one_hot:
                    action = one_hot_vector(action, self.one_hot_dim)
                demo.append((np.expand_dims(obs, axis=0), action, reward, done))
                if done:
                    break
                obs = next_obs

            demo_return = self._compute_discounted_return(demo)
            if demo_return > best_return:
                best_return = demo_return
                best_demo = demo

        return best_demo

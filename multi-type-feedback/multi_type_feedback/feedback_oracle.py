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
        self.action_one_hot = isinstance(
            self.environment.action_space, gym.spaces.Discrete
        )
        if self.action_one_hot:
            self.one_hot_dim = self.environment.action_space.n

        # Load and process reference data for calibration
        self.initialize_calibration(reference_data_path)

    def initialize_calibration(self, reference_data_path: str):
        """Initialize calibration data from pre-computed reference trajectories."""
        with open(reference_data_path, "rb") as f:
            reference_data = pickle.load(f)
    
        # ---- Evaluative feedback calibration (ECDF over optimality gaps) ----
        self.reference_opt_gaps = np.asarray(reference_data["opt_gaps"], dtype=float)
        self._opt_gaps_sorted = np.sort(self.reference_opt_gaps)
    
        # ---- Build state-action feature matrix (flatten obs; ensure action >= 1D) ----
        def _flatten_sa(step):
            obs = np.asarray(step[0])
            obs = np.squeeze(obs)          # drop singletons, e.g. (1,H,W)->(H,W)
            obs_flat = obs.reshape(-1)     # ALWAYS 1-D
            act = np.asarray(step[1])
            act_flat = np.atleast_1d(act)  # scalar or vector → 1-D
            return np.concatenate((obs_flat, act_flat), axis=0)
    
        states_actions_list, rewards_list = [], []
    
        for segment in reference_data["segments"]:
            for step in segment:
                states_actions_list.append(_flatten_sa(step))
                rewards_list.append(step[2])
    
        if not states_actions_list:
            raise ValueError("No steps found in reference_data['segments'] to calibrate.")
    
        states_actions = np.asarray(states_actions_list, dtype=float)
        rewards = np.asarray(rewards_list, dtype=float)
    
        # ---- Standardize features for clustering ----
        self._sa_mean = states_actions.mean(axis=0)
        self._sa_std = states_actions.std(axis=0)
        self._sa_std[self._sa_std == 0] = 1.0  # avoid division by zero
        states_actions_std = (states_actions - self._sa_mean) / self._sa_std
    
        # ---- Fit clustering ----
        batch_size = max(1, min(1000, len(states_actions_std) // 100))
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=batch_size, random_state=42
        )
        self.kmeans.fit(states_actions_std)
    
        # ---- Compute per-cluster averages and store representatives ----
        cluster_assignments = self.kmeans.predict(states_actions_std)
    
        self.cluster_representatives = []  # de-standardized centers (raw feature space)
        self.cluster_rewards = []          # avg per-step reward for cluster
        self._cluster_centers_std = []     # centers in standardized space (for NN queries)
    
        for i in range(self.n_clusters):
            mask = cluster_assignments == i
            if np.any(mask):
                center_std = self.kmeans.cluster_centers_[i]
                center = center_std * self._sa_std + self._sa_mean
                avg_reward = float(rewards[mask].mean())
                self.cluster_representatives.append(center)
                self.cluster_rewards.append(avg_reward)
                self._cluster_centers_std.append(center_std)
    
        self.cluster_representatives = np.asarray(self.cluster_representatives, dtype=float)
        self.cluster_rewards = np.asarray(self.cluster_rewards, dtype=float)
        self._cluster_centers_std = np.asarray(self._cluster_centers_std, dtype=float)
    
        if self.cluster_representatives.size == 0:
            raise ValueError("Clustering produced no non-empty clusters. Check reference data.")


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
                raise ValueError(
                    f"{feedback_type} feedback requires a single trajectory"
                )

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
                raise ValueError(
                    f"{feedback_type} feedback requires a pair of trajectories"
                )

            trajectory1, trajectory2 = trajectory_data

            if feedback_type == "comparative":
                return self.get_comparative_feedback(trajectory1, trajectory2)
            elif feedback_type == "descriptive_preference":
                return self.get_descriptive_preference_feedback(
                    trajectory1, trajectory2
                )

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
            actions = torch.cat(
                [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
            )
            mask = torch.cat([mask, torch.zeros(pad_size, 1)], dim=0)

        # Calculate rating via empirical CDF on reference distribution
        opt_gap = -self._compute_discounted_return(trajectory)

        if self.noise_level > 0:
            opt_gap += np.random.normal(
                0, self.noise_level * np.std(self.reference_opt_gaps)
            )

        # Fraction of reference opt_gaps <= current (ECDF), map better (smaller) gaps to higher ratings
        cdf = np.searchsorted(self._opt_gaps_sorted, opt_gap, side="right") / max(
            1, self._opt_gaps_sorted.size
        )
        rating = 1.0 - float(cdf)
        rating = float(np.clip(rating, 0.0, 1.0))

        return (obs, actions, mask), rating

    def get_comparative_feedback(
        self,
        trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ):
        return1 = self._compute_discounted_return(trajectory1)
        return2 = self._compute_discounted_return(trajectory2)

        # trajectory 1
        trajectory1_obs = torch.vstack(
            [torch.as_tensor(p[0]).float() for p in trajectory1]
        )
        trajectory1_actions = torch.vstack(
            [torch.as_tensor(p[1]).float() for p in trajectory1]
        )

        # trajectory 2
        trajectory2_obs = torch.vstack(
            [torch.as_tensor(p[0]).float() for p in trajectory2]
        )
        trajectory2_actions = torch.vstack(
            [torch.as_tensor(p[1]).float() for p in trajectory2]
        )

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
    ) -> Tuple[
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], int
    ]:
        """Return demonstration and random trajectory pair with preference label 1."""
        demo = self._get_best_demonstration(initial_state)
        random_trajectory = self.get_random_trajectory()

        # Convert demo to tensor format
        obs_demo = torch.vstack([torch.as_tensor(p[0]).float() for p in demo])
        actions_demo = torch.vstack([torch.as_tensor(p[1]).float() for p in demo])

        # Convert random trajectory to tensor format
        obs_rand = torch.vstack(
            [torch.as_tensor(p[0]).float() for p in random_trajectory]
        )
        actions_rand = torch.vstack(
            [torch.as_tensor(p[1]).float() for p in random_trajectory]
        )

        # Pad both trajectories if necessary
        mask_demo = torch.ones(len(demo)).unsqueeze(-1)
        if len(demo) < self.segment_len:
            pad_size = self.segment_len - len(demo)
            obs_demo = torch.cat(
                [obs_demo, torch.zeros(pad_size, *obs_demo.shape[1:])], dim=0
            )
            actions_demo = torch.cat(
                [actions_demo, torch.zeros(pad_size, *actions_demo.shape[1:])], dim=0
            )
            mask_demo = torch.cat([mask_demo, torch.zeros(pad_size, 1)], dim=0)

        mask_rand = torch.ones(len(random_trajectory)).unsqueeze(-1)
        if len(random_trajectory) < self.segment_len:
            pad_size = self.segment_len - len(random_trajectory)
            obs_rand = torch.cat(
                [obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0
            )
            actions_rand = torch.cat(
                [actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])], dim=0
            )
            mask_rand = torch.cat([mask_rand, torch.zeros(pad_size, 1)], dim=0)
        
        return (
            (obs_rand, actions_rand, mask_rand),
            (obs_demo, actions_demo, mask_demo),
        ), 1

    def get_corrective_feedback(
        self, trajectory, initial_state
    ) -> Tuple[
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], int
    ]:
        """Return original and corrected trajectory pair with preference label 1."""
        trajectory_return = self._compute_discounted_return(trajectory)
        expert_demo = self._get_best_demonstration(initial_state)
        expert_return = self._compute_discounted_return(expert_demo)

        if self.noise_level > 0:
            expert_return += np.random.normal(0, self.noise_level * abs(expert_return))
            trajectory_return += np.random.normal(
                0, self.noise_level * abs(trajectory_return)
            )

        # Convert trajectories to tensor format
        obs_orig = torch.vstack([torch.as_tensor(p[0]).float() for p in trajectory])
        actions_orig = torch.vstack([torch.as_tensor(p[1]).float() for p in trajectory])

        obs_expert = torch.vstack([torch.as_tensor(p[0]).float() for p in expert_demo])
        actions_expert = torch.vstack(
            [torch.as_tensor(p[1]).float() for p in expert_demo]
        )

        # Pad if necessary
        mask_traj = torch.ones(len(trajectory)).unsqueeze(-1)
        if len(trajectory) < self.segment_len:
            pad_size = self.segment_len - len(trajectory)
            obs_orig = torch.cat(
                [obs_orig, torch.zeros(pad_size, *obs_orig.shape[1:])], dim=0
            )
            actions_orig = torch.cat(
                [actions_orig, torch.zeros(pad_size, *actions_orig.shape[1:])], dim=0
            )
            mask_traj = torch.cat([mask_traj, torch.zeros(pad_size, 1)], dim=0)

        mask_demo = torch.ones(len(expert_demo)).unsqueeze(-1)
        if len(expert_demo) < self.segment_len:
            pad_size = self.segment_len - len(expert_demo)
            obs_expert = torch.cat(
                [obs_expert, torch.zeros(pad_size, *obs_expert.shape[1:])], dim=0
            )
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
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
        """Return the most similar cluster representative and its average reward."""
        # Build flattened state-action features
        states_actions = np.array(
            [
                np.concatenate(
                    (np.squeeze(step[0]).reshape(-1), np.atleast_1d(step[1]))
                )
                for step in trajectory
            ],
            dtype=float,
        )
        avg_state_action = np.mean(states_actions, axis=0)
    
        # Nearest cluster in standardized space
        avg_std = (avg_state_action - self._sa_mean) / self._sa_std
        distances = cdist([avg_std], self._cluster_centers_std)
        closest_cluster = int(np.argmin(distances))
    
        # Representative (raw space) and reward
        representative = self.cluster_representatives[closest_cluster]
        reward = float(self.cluster_rewards[closest_cluster])
    
        if self.noise_level > 0:
            reward += np.random.normal(0, self.noise_level * np.std(self.cluster_rewards))
    
        # Split representative back into (state, action) using flattened obs dim
        obs_example = np.asarray(trajectory[0][0])
        obs_dim = int(np.prod(np.squeeze(obs_example).shape))
    
        state_rep = torch.as_tensor(representative[:obs_dim]).unsqueeze(0).float()
        action_rep = torch.as_tensor(representative[obs_dim:]).unsqueeze(0).float()
        mask = torch.ones(1).unsqueeze(-1)  # for compatibility
    
        return (state_rep, action_rep, mask), reward
    
    
    def get_descriptive_preference_feedback(
        self,
        trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ) -> Tuple[
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
        """Compare two trajectories based on their closest cluster representatives."""
        def _avg_sa(traj):
            X = np.array(
                [
                    np.concatenate(
                        (np.squeeze(step[0]).reshape(-1), np.atleast_1d(step[1]))
                    )
                    for step in traj
                ],
                dtype=float,
            )
            return X.mean(axis=0)
    
        avg1 = _avg_sa(trajectory1)
        avg2 = _avg_sa(trajectory2)
    
        avg1_std = (avg1 - self._sa_mean) / self._sa_std
        avg2_std = (avg2 - self._sa_mean) / self._sa_std
    
        d1 = cdist([avg1_std], self._cluster_centers_std)
        d2 = cdist([avg2_std], self._cluster_centers_std)
    
        c1 = int(np.argmin(d1))
        c2 = int(np.argmin(d2))
    
        rep1 = self.cluster_representatives[c1]
        rep2 = self.cluster_representatives[c2]
    
        r1 = float(self.cluster_rewards[c1])
        r2 = float(self.cluster_rewards[c2])
    
        if self.noise_level > 0:
            noise_scale = self.noise_level * np.std(self.cluster_rewards)
            r1 += np.random.normal(0, noise_scale)
            r2 += np.random.normal(0, noise_scale)
    
        # determine flattened obs_dim for splitting
        obs_example = np.asarray(trajectory1[0][0])
        obs_dim = int(np.prod(np.squeeze(obs_example).shape))
    
        def _to_tensors(rep):
            s = torch.as_tensor(rep[:obs_dim]).unsqueeze(0).float()
            a = torch.as_tensor(rep[obs_dim:]).unsqueeze(0).float()
            m = torch.ones(1).unsqueeze(-1)
            return (s, a, m)
    
        pair = (_to_tensors(rep1), _to_tensors(rep2))
        if r1 > r2:
            # Return (worse, better), label=1 (preference for second)
            return (pair[0], pair[1]), 1
        else:
            return (pair[1], pair[0]), 1
    

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

        for exp_model_index, (expert_model, exp_norm_env) in enumerate(
            self.expert_models
        ):
            self.environment.reset()
            obs = self.environment.load_state(initial_state)
            demo = []

            for _ in range(self.segment_len):
                action, _ = expert_model.predict(
                    exp_norm_env.normalize_obs(obs) if exp_norm_env else obs,
                    deterministic=True,
                )
                next_obs, reward, terminated, truncated, _ = self.environment.step(
                    action
                )
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
import os
import pickle
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import truncnorm, uniform
from torch.utils.data import Dataset

from multi_type_feedback.datatypes import FeedbackData, FeedbackType, SegmentT

def truncated_uniform_vectorized(mean, width, low=0, upp=9):
    # Handle scalar inputs
    scalar_input = np.isscalar(mean) and np.isscalar(width)
    mean = np.atleast_1d(mean)
    width = np.atleast_1d(width)

    # Calculate the bounds of the uniform distribution
    lower = mean - width / 2
    upper = mean + width / 2

    # Clip the bounds to [low, upp]
    lower = np.clip(lower, low, upp)
    upper = np.clip(upper, low, upp)

    # Generate random values
    r = np.random.uniform(size=mean.shape)

    # Calculate the result
    result = lower + r * (upper - lower)

    return result[0] if scalar_input else result


def truncated_gaussian_vectorized(mean, width, low=0, upp=9, min_width=1e-6):
    # Handle scalar inputs
    scalar_input = np.isscalar(mean) and np.isscalar(width)
    mean = np.atleast_1d(mean)
    width = np.atleast_1d(width)

    # Ensure width is positive to avoid numerical issues
    width = np.maximum(width, min_width)

    # Calculate the bounds of the distribution
    lower = np.maximum(mean - width / 2, low)
    upper = np.minimum(mean + width / 2, upp)

    # Calculate parameters for truncated normal distribution
    a = (lower - mean) / (width / 4)  # 4 sigma range
    b = (upper - mean) / (width / 4)

    # Generate samples from truncated normal distribution
    result = truncnorm.rvs(a, b, loc=mean, scale=width / 4, size=mean.shape)

    return result[0] if scalar_input else result


def discounted_sum_numpy(rewards, gamma):
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))


def _chunk_sequence(sequence: List[Tuple], chunk_size: int) -> List[List[Tuple]]:
    """Split a trajectory into chunks of at most chunk_size elements."""
    if not sequence or chunk_size <= 0:
        return [sequence] if sequence else []
    return [sequence[i : i + chunk_size] for i in range(0, len(sequence), chunk_size)]


def _expand_feedback_sequences(
    feedback_data: Dict,
    feedback_type: FeedbackType,
    segment_len: int,
) -> Dict:
    """Ensure all trajectory-based feedback operates on segments of length <= segment_len."""
    if feedback_data is None:
        return feedback_data

    # Convert to dict copy to avoid mutating callers unexpectedly
    data = dict(feedback_data)

    if feedback_type == "evaluative":
        segments = data.get("segments", [])
        ratings = data.get("ratings", [])
        opt_gaps = data.get("opt_gaps", [])
        expanded_segments = []
        expanded_ratings = []
        expanded_opt_gaps = []

        for idx, seg in enumerate(segments):
            chunks = _chunk_sequence(seg, segment_len)
            rating = ratings[idx] if idx < len(ratings) else 0.0
            opt_gap = opt_gaps[idx] if idx < len(opt_gaps) else 0.0
            for chunk in chunks:
                expanded_segments.append(chunk)
                expanded_ratings.append(rating)
                expanded_opt_gaps.append(opt_gap)

        data["segments"] = expanded_segments
        data["ratings"] = expanded_ratings
        data["opt_gaps"] = expanded_opt_gaps

    elif feedback_type == "comparative":
        segments = data.get("segments", [])
        opt_gaps = data.get("opt_gaps", [])
        preferences = data.get("preferences", [])

        expanded_segments = []
        expanded_opt_gaps = []
        index_mapping: Dict[int, List[int]] = {}

        for idx, seg in enumerate(segments):
            chunks = _chunk_sequence(seg, segment_len)
            opt_gap = opt_gaps[idx] if idx < len(opt_gaps) else 0.0
            new_indices = []
            for chunk in chunks:
                new_index = len(expanded_segments)
                expanded_segments.append(chunk)
                expanded_opt_gaps.append(opt_gap)
                new_indices.append(new_index)
            index_mapping[idx] = new_indices

        expanded_preferences = []
        for pref_entry in preferences:
            if len(pref_entry) < 3:
                continue
            idx1, idx2, pref_value = pref_entry
            new_list_1 = index_mapping.get(idx1, [])
            new_list_2 = index_mapping.get(idx2, [])
            pair_count = min(len(new_list_1), len(new_list_2))
            if pair_count == 0:
                continue
            if len(new_list_1) != len(new_list_2):
                print(
                    f"Warning: Preference segments have mismatched lengths "
                    f"(idx1={idx1}, idx2={idx2}); truncating to shortest."
                )
            for offset in range(pair_count):
                expanded_preferences.append([new_list_1[offset], new_list_2[offset], pref_value])

        data["segments"] = expanded_segments
        data["opt_gaps"] = expanded_opt_gaps
        data["preferences"] = expanded_preferences

    elif feedback_type == "demonstrative":
        demos = data.get("demos", [])
        expanded_demos = []
        for demo in demos:
            expanded_demos.extend(_chunk_sequence(demo, segment_len))
        data["demos"] = expanded_demos

    elif feedback_type == "corrective":
        corrections = data.get("corrections", [])
        expanded_corrections = []

        for correction in corrections:
            if not correction or len(correction) < 2:
                continue
            ref, corr = correction[0], correction[1]
            ref_chunks = _chunk_sequence(ref, segment_len)
            corr_chunks = _chunk_sequence(corr, segment_len)
            pair_count = min(len(ref_chunks), len(corr_chunks))
            if pair_count == 0:
                continue
            if len(ref_chunks) != len(corr_chunks):
                print("Warning: Correction segments have different lengths; truncating to shortest.")
            for offset in range(pair_count):
                expanded_corrections.append([ref_chunks[offset], corr_chunks[offset]])
        data["corrections"] = expanded_corrections

    return data


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(
        self,
        feedback_data: FeedbackData,
        feedback_type: FeedbackType,
        n_feedback: int,
        env_name: str = "",
        noise_level: float = 0.0,  # depending on the feedback, we use different types of noise (e.g. flip the preference, add noise to the rating or description)
        segment_len: int = 50,
        env=None,
        seed: int = 1234,
        zero_tolerance: float = 1e6,
        discount_factor: float = 0.99,
        stratifier=None,
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        self.targets: Union[
            List[SegmentT],
            List[NDArray],
            Tuple[SegmentT, SegmentT, SegmentT],
            Tuple[NDArray, NDArray, NDArray],
        ] = []
        self.preds: List[int] = []
        self.ranks: List[float] = []  # For RT-rank loss
        self.partition_ids: List[int] = []  # For stratification

        feedback_data = _expand_feedback_sequences(feedback_data, feedback_type, segment_len)

        if not feedback_data:
            self.targets = torch.empty((0, segment_len if segment_len else 1))
            self.preds = torch.empty(0)
            return self.targets, self.preds

        if feedback_type == "evaluative":
            for seg in feedback_data["segments"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in seg])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in seg])
                
                # Create mask for valid timesteps
                original_len = obs.size(0)
                mask = torch.ones(original_len, 1)

                # Pad both trajectories to the maximum length
                if original_len < segment_len:
                    pad_size = segment_len - original_len
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                    mask = torch.cat([mask, torch.zeros(pad_size, 1)], dim=0)

                self.targets.append((obs, actions, mask))

            self.preds = feedback_data["ratings"]
            # add noise to the ratings
            if noise_level > 0:
                # apply noise, accounting for clipping at the borders [0,9]
                self.preds = truncated_gaussian_vectorized(
                    mean=self.preds,
                    width=noise_level * 10 * np.ones_like(self.preds),
                    low=0,
                    upp=9,
                )
            
            # Initialize empty ranks and partition_ids for non-comparative feedback
            self.ranks = [0.0] * len(self.targets)
            self.partition_ids = [0] * len(self.targets)
        elif feedback_type == "comparative":
            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            ref_diff = np.abs(rews_max - rews_min)

            # Collect all examples for stratification
            all_examples = []
            flipped = 0
            for comp in feedback_data["preferences"]:
                # seg 1
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp[0]]])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp[0]]])
                
                # seg 2
                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp[1]]])
                actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp[1]]])

                # Create masks for valid timesteps
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)
                mask = torch.ones(len_obs, 1)
                mask2 = torch.ones(len_obs2, 1)

                # Calculate reward difference for synthetic ranking
                rew1 = -feedback_data["opt_gaps"][comp[0]]
                rew2 = -feedback_data["opt_gaps"][comp[1]]
                reward_diff = abs(rew1 - rew2)
                
                # Synthetic rank based on negative reward difference (higher diff = stronger preference)
                synthetic_rank = -reward_diff
                
                # Create example metadata for stratification
                trajectory_length = max(len_obs, len_obs2)
                example_metadata = {
                    "rank": synthetic_rank,
                    "trajectory_length": trajectory_length,
                    "evaluator": "default",  # Could be extended to use actual evaluator info
                }
                all_examples.append(example_metadata)

                # Pad both trajectories to the maximum length, necessary for batching with data loader
                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                    mask = torch.cat([mask, torch.zeros(pad_size, 1)], dim=0)
                    
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0)
                    mask2 = torch.cat([mask2, torch.zeros(pad_size, 1)], dim=0)

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1_noisy = truncated_gaussian_vectorized(
                        mean=np.array(rew1),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )
                    rew2_noisy = truncated_gaussian_vectorized(
                        mean=np.array(rew2),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )

                    # put the "chosen" segment first
                    if rew1_noisy > rew2_noisy:
                        self.targets.append(((obs, actions, mask), (obs2, actions2, mask2)))
                    else:
                        self.targets.append(((obs2, actions2, mask2), (obs, actions, mask)))
                        flipped += 1
                    self.preds.append(0)
                else:
                    self.targets.append(((obs, actions, mask), (obs2, actions2, mask2)))
                    self.preds.append(0)
                
                self.ranks.append(synthetic_rank)
            
            # Compute stratification partitions if stratifier is provided
            if stratifier is not None and all_examples:
                rng = np.random.default_rng(seed)
                self.partition_ids = stratifier.compute_partitions(all_examples, rng)
                print(f"Computed {len(set(self.partition_ids))} partitions for RT-rank loss")
            else:
                # Default: all examples in same partition
                self.partition_ids = [0] * len(self.targets)

        elif feedback_type == "demonstrative":
            with open(os.path.join("multi-type-feedback", "samples", f"random_{env_name}.pkl"), "rb") as random_file:
                random_data = pickle.load(random_file)

            for demo in feedback_data["demos"]:
                obs = np.vstack([p[0] for p in demo])
                actions = np.vstack([p[1] for p in demo])

                if noise_level > 0.0:
                    # Calculate statistics across all data points, keeping the feature dimensions
                    obs_min, obs_max, obs_std = (
                        np.min(obs, axis=0),
                        np.max(obs, axis=0),
                        np.std(obs, axis=0),
                    )
                    non_zero_obs_std = obs_std > zero_tolerance

                    acts_min, acts_max, acts_std = (
                        np.min(actions, axis=0),
                        np.max(actions, axis=0),
                        np.std(actions, axis=0),
                    )
                    non_zero_acts_std = acts_std > zero_tolerance

                    # Process each batch separately
                    noisy_obs = []
                    noisy_actions = []

                    for i in range(obs.shape[0]):
                        obs_for_noise = obs[i]
                        if np.any(non_zero_obs_std):
                            obs_for_noise[:, non_zero_obs_std] = (
                                truncated_gaussian_vectorized(
                                    mean=obs_for_noise[:, non_zero_obs_std],
                                    width=np.array(noise_level) * obs_std[non_zero_obs_std],
                                    low=obs_min[non_zero_obs_std],
                                    upp=obs_max[non_zero_obs_std],
                                )
                            )
                        noisy_obs.append(obs_for_noise)

                        acts_for_noise = actions[i]
                        if np.any(non_zero_acts_std):
                            acts_for_noise[:, non_zero_acts_std] = (
                                truncated_gaussian_vectorized(
                                    mean=acts_for_noise[:, non_zero_acts_std],
                                    width=np.array(noise_level) * acts_std[non_zero_acts_std],
                                    low=acts_min[non_zero_acts_std],
                                    upp=acts_max[non_zero_acts_std],
                                )
                            )
                        noisy_actions.append(acts_for_noise)

                    obs = np.stack(noisy_obs)
                    actions = np.stack(noisy_actions)

                obs = torch.as_tensor(obs).float()
                actions = torch.as_tensor(actions).float()
                
                # Create mask for demo
                original_len_demo = obs.size(0)
                mask_demo = torch.ones(original_len_demo, 1)

                # just use a random segment as the opposite
                import random
                rand_index = random.randrange(0, len(random_data["segments"]))
                obs_rand = torch.vstack([torch.as_tensor(p[0]).float() for p in random_data["segments"][rand_index]])
                actions_rand = torch.vstack([torch.as_tensor(p[1]).float() for p in random_data["segments"][rand_index]])
                
                # Create mask for random segment
                original_len_rand = obs_rand.size(0)
                mask_rand = torch.ones(original_len_rand, 1)

                # Pad both trajectories to the maximum length
                if original_len_demo < segment_len:
                    pad_size = segment_len - original_len_demo
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                    mask_demo = torch.cat([mask_demo, torch.zeros(pad_size, 1)], dim=0)
                    
                if original_len_rand < segment_len:
                    pad_size = segment_len - original_len_rand
                    obs_rand = torch.cat([obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0)
                    actions_rand = torch.cat([actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])], dim=0)
                    mask_rand = torch.cat([mask_rand, torch.zeros(pad_size, 1)], dim=0)

                self.targets.append(((obs_rand, actions_rand, mask_rand), (obs, actions, mask_demo)))
                self.preds.append(1)  # assume that the demonstration is optimal

            # Initialize empty ranks and partition_ids for demonstrative feedback
            self.ranks = [0.0] * len(self.targets)
            self.partition_ids = [0] * len(self.targets)

        elif feedback_type == "corrective":
            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            rew_diff = np.abs(rews_max - rews_min)
            gamma = discount_factors[env_name]

            flipped = 0
            for comp in feedback_data["corrections"]:
                obs1 = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[0]])
                actions1 = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[0]])

                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[1]])
                actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[1]])

                # Create masks for valid timesteps
                original_len1 = obs1.size(0)
                original_len2 = obs2.size(0)
                mask1 = torch.ones(original_len1, 1)
                mask2 = torch.ones(original_len2, 1)

                # Pad both trajectories to the maximum length
                if original_len1 < segment_len:
                    pad_size = segment_len - original_len1
                    obs1 = torch.cat([obs1, torch.zeros(pad_size, *obs1.shape[1:])], dim=0)
                    actions1 = torch.cat([actions1, torch.zeros(pad_size, *actions1.shape[1:])], dim=0)
                    mask1 = torch.cat([mask1, torch.zeros(pad_size, 1)], dim=0)
                    
                if original_len2 < segment_len:
                    pad_size = segment_len - original_len2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0)
                    mask2 = torch.cat([mask2, torch.zeros(pad_size, 1)], dim=0)

                # add noise and recompute preferences
                if noise_level > 0.0:
                    rews1 = discounted_sum_numpy(np.array([p[2] for p in comp[0]]), gamma)
                    rews2 = discounted_sum_numpy(np.array([p[2] for p in comp[1]]), gamma)

                    rew1 = truncated_gaussian_vectorized(
                        mean=rews1,
                        width=np.array(noise_level) * rew_diff,
                        low=min(rews_min, rews1),
                        upp=max(rews_max, rews1),
                    ).item()
                    rew2 = truncated_gaussian_vectorized(
                        mean=rews2,
                        width=np.array(noise_level) * rew_diff,
                        low=min(rews_min, rews2),
                        upp=max(rews_max, rews2),
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs1, actions1, mask1), (obs2, actions2, mask2)))
                    else:
                        self.targets.append(((obs2, actions2, mask2), (obs1, actions1, mask1)))
                        flipped += 1
                    self.preds.append(1)
                else:
                    self.targets.append(((obs1, actions1, mask1), (obs2, actions2, mask2)))
                    self.preds.append(1)
            
            # Initialize empty ranks and partition_ids for corrective feedback
            self.ranks = [0.0] * len(self.targets)
            self.partition_ids = [0] * len(self.targets)

        elif feedback_type == "descriptive":
            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)

            for cluster_representative in feedback_data["description"]:
                obs = torch.as_tensor(cluster_representative[0]).unsqueeze(0).float()
                actions = torch.as_tensor(cluster_representative[1]).unsqueeze(0).float()
                mask = torch.ones(1, 1)  # Single timestep mask
                
                self.targets.append((obs, actions, mask))

                if noise_level > 0.0:
                    rew = truncated_gaussian_vectorized(
                        mean=np.array(cluster_representative[2]),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
                    )
                    self.preds.append(rew.item())
                else:
                    self.preds.append(cluster_representative[2])
            
            # Initialize empty ranks and partition_ids for descriptive feedback
            self.ranks = [0.0] * len(self.targets)
            self.partition_ids = [0] * len(self.targets)

        elif feedback_type == "descriptive_preference":
            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)

            flipped = 0
            for cpref in feedback_data["description_preference"]:
                idx_1 = cpref[0]
                idx_2 = cpref[1]

                # cluster 1
                obs1 = torch.as_tensor(feedback_data["description"][idx_1][0]).unsqueeze(0).float()
                actions1 = torch.as_tensor(feedback_data["description"][idx_1][1]).unsqueeze(0).float()
                mask1 = torch.ones(1, 1)

                # cluster 2
                obs2 = torch.as_tensor(feedback_data["description"][idx_2][0]).unsqueeze(0).float()
                actions2 = torch.as_tensor(feedback_data["description"][idx_2][1]).unsqueeze(0).float()
                mask2 = torch.ones(1, 1)

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = feedback_data["description"][idx_1][2]
                    rew2 = feedback_data["description"][idx_2][2]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
                    ).item()
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs1, actions1, mask1), (obs2, actions2, mask2)))
                    else:
                        self.targets.append(((obs2, actions2, mask2), (obs1, actions1, mask1)))
                        flipped += 1
                    self.preds.append(cpref[2])
                else:
                    self.targets.append(((obs1, actions1, mask1), (obs2, actions2, mask2)))
                    self.preds.append(cpref[2])
            
            # Initialize empty ranks and partition_ids for descriptive_preference feedback
            self.ranks = [0.0] * len(self.targets)
            self.partition_ids = [0] * len(self.targets)
        else:
            raise NotImplementedError("Dataset not implemented for this feedback type.")

        print("Dataset loaded")
        print(f"N TARGETS AVAILABLE: {len(self.targets)}, N_FEEDBACK: {n_feedback}")

        if n_feedback != -1 and n_feedback < len(self.targets):
            # is a bit inefficient as we first collected the entire dataset..but we just have to do it once
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.targets), size=n_feedback, replace=False)

            self.targets = [self.targets[i] for i in indices]
            self.preds = [self.preds[i] for i in indices]

    def __len__(self):
        """Return size of dataset."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return item with given index."""
        if hasattr(self, 'ranks') and len(self.ranks) > 0:
            # Return additional data for RT-rank loss
            return (self.targets[index], self.preds[index], 
                   self.ranks[index], self.partition_ids[index])
        else:
            return self.targets[index], self.preds[index]


class LoadFeedbackDataset(FeedbackDataset):
    """Load feedback dataset from file."""

    def __init__(
        self,
        dataset_path: str,
        feedback_type: FeedbackType,
        n_feedback: int,
        env_name: str = "",
        noise_level: float = 0.0,
        segment_len: int = 50,
        env=None,
        seed: int = 1234,
        discount_factor: float = 0.99,
        stratifier=None,
    ):

        with open(dataset_path, "rb") as feedback_file:
            feedback_data: FeedbackData = pickle.load(feedback_file)

        super().__init__(
            feedback_data,
            feedback_type,
            n_feedback,
            env_name,
            noise_level,
            segment_len,
            env,
            seed,
            discount_factor=discount_factor,
            stratifier=stratifier,
        )


class BufferDataset(Dataset):

    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        """Return size of dataset."""
        return len(self.buffer)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.buffer[index]


FEEDBACK_TYPE_TO_KEY = {
    "evaluative": "segments",
    "comparative": "segments",
    "demonstrative": "demos",
    "corrective": "corrections",
    "descriptive": "description",
    "descriptive_preference": "description_preference",
}


def load_flat_buffer_into_feedback_dataset(
    feedback_buffer: List[SegmentT], feedback_type: FeedbackType
) -> dict:
    """Maps feedback buffer to the appropriate feedback data structure."""
    if feedback_type not in FEEDBACK_TYPE_TO_KEY:
        raise NotImplementedError(f"Feedback type {feedback_type} not implemented.")
    return {FEEDBACK_TYPE_TO_KEY[feedback_type]: feedback_buffer}

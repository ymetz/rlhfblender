import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import truncnorm
from torch.utils.data import Dataset

from multi_type_feedback.datatypes import FeedbackData, FeedbackType, SegmentT
from multi_type_feedback.utils import discount_factors


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
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        self.targets: Union[
            List[SegmentT],
            List[NDArray],
            Tuple[SegmentT, SegmentT],
            Tuple[NDArray, NDArray],
        ] = []
        self.preds: List[int] = []

        if not feedback_data:
            self.targets = torch.empty((0, segment_len if segment_len else 1))
            self.preds = torch.empty(0)
            return self.targets, self.preds

        if feedback_type == "evaluative":
            for seg in feedback_data["segments"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in seg])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in seg])

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )

                self.targets.append((obs, actions))

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
        elif feedback_type == "comparative":

            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            ref_diff = np.abs(rews_max - rews_min)

            flipped = 0
            for comp in feedback_data["preferences"]:
                # seg 1
                obs = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in feedback_data["segments"][comp[0]]
                    ]
                )
                actions = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in feedback_data["segments"][comp[0]]
                    ]
                )

                # seg 2
                obs2 = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in feedback_data["segments"][comp[1]]
                    ]
                )
                actions2 = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in feedback_data["segments"][comp[1]]
                    ]
                )

                # Pad both trajectories to the maximum length, necessary for batching with data loader
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat(
                        [obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0
                    )
                    actions2 = torch.cat(
                        [actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0
                    )

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = -feedback_data["opt_gaps"][comp[0]]
                    rew2 = -feedback_data["opt_gaps"][comp[1]]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )

                    if rew2 > rew1:
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(comp[2])
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(comp[2])

        elif feedback_type == "demonstrative":

            with open(
                os.path.join("samples", f"random_{env_name}.pkl"), "rb"
            ) as random_file:
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

                    # TODO: Check if this for loop is actually necessary...shouldn't it just work as a batch
                    for i in range(obs.shape[0]):
                        # Add noise to each batch independently

                        obs_for_noise = obs[i]
                        if np.any(non_zero_obs_std):
                            obs_for_noise[:, non_zero_obs_std] = (
                                truncated_gaussian_vectorized(
                                    mean=obs_for_noise[:, non_zero_obs_std],
                                    width=np.array(noise_level)
                                    * obs_std[non_zero_obs_std],
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
                                    width=np.array(noise_level)
                                    * acts_std[non_zero_acts_std],
                                    low=acts_min[non_zero_acts_std],
                                    upp=acts_max[non_zero_acts_std],
                                )
                            )
                        noisy_actions.append(acts_for_noise)

                    obs = np.stack(noisy_obs)
                    actions = np.stack(noisy_actions)

                obs = torch.as_tensor(obs).float()
                actions = torch.as_tensor(actions).float()

                # just use a random segment as the opposite
                rand_index = random.randrange(0, len(random_data["segments"]))
                obs_rand = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in random_data["segments"][rand_index]
                    ]
                )
                actions_rand = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in random_data["segments"][rand_index]
                    ]
                )

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs_rand = obs_rand.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs_rand < segment_len:
                    pad_size = segment_len - len_obs_rand
                    obs_rand = torch.cat(
                        [obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0
                    )
                    actions_rand = torch.cat(
                        [actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])],
                        dim=0,
                    )

                self.targets.append(((obs_rand, actions_rand), (obs, actions)))
                self.preds.append(
                    1
                )  # assume that the demonstration is optimal, maybe add confidence value (based on regret)
        elif feedback_type == "corrective":

            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            rew_diff = np.abs(rews_max - rews_min)
            gamma = discount_factors[env_name]

            flipped = 0
            for comp in feedback_data["corrections"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[0]])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[0]])

                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[1]])
                actions2 = torch.vstack(
                    [torch.as_tensor(p[1]).float() for p in comp[1]]
                )

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat(
                        [obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0
                    )
                    actions2 = torch.cat(
                        [actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0
                    )

                # add noise and recompute preferences
                if noise_level > 0.0:
                    rews1 = discounted_sum_numpy(
                        np.array([p[2] for p in comp[0]]), gamma
                    )
                    rews2 = discounted_sum_numpy(
                        np.array([p[2] for p in comp[1]]), gamma
                    )

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
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(1)
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(1)
        elif feedback_type == "descriptive":
            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)

            for cluster_representative in feedback_data["description"]:
                self.targets.append(
                    (
                        torch.as_tensor(cluster_representative[0]).unsqueeze(0).float(),
                        torch.as_tensor(cluster_representative[1]).unsqueeze(0).float(),
                    )
                )

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
        elif feedback_type == "descriptive_preference":
            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)

            flipped = 0
            for cpref in feedback_data["description_preference"]:
                idx_1 = cpref[0]

                # cluster 1
                obs = (
                    torch.as_tensor(feedback_data["description"][idx_1][0])
                    .unsqueeze(0)
                    .float()
                )
                actions = (
                    torch.as_tensor(feedback_data["description"][idx_1][1])
                    .unsqueeze(0)
                    .float()
                )

                idx_2 = cpref[1]

                # cluster 2
                obs2 = (
                    torch.as_tensor(feedback_data["description"][idx_2][0])
                    .unsqueeze(0)
                    .float()
                )
                actions2 = (
                    torch.as_tensor(feedback_data["description"][idx_2][1])
                    .unsqueeze(0)
                    .float()
                )

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
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(cpref[2])
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(cpref[2])
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
from dataclasses import dataclass

import numpy as np


@dataclass
class RecordedEpisodesContainer:
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    actions: np.ndarray
    infos: np.ndarray
    renders: np.ndarray
    features: np.ndarray
    probs: np.ndarray
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    additional_metrics: np.ndarray

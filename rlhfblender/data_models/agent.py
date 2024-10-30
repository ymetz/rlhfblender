from abc import ABC
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class BaseAgent:
    def __init__(self, observation_space, action_space, env, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_vec_env = isinstance(env, VecEnv)

    def act(self, observation):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def additional_outputs(self, observation, action, output_list=None) -> Optional[Dict]:
        raise NotImplementedError


class TrainedAgent(BaseAgent, ABC):
    def __init__(self, observation_space, action_space, path, env, device="auto", **kwargs):
        super().__init__(observation_space, action_space, env, **kwargs)
        self.path = path
        self.device = device

    def act(self, observation):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def extract_features(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def additional_outputs(
        self, observation: np.ndarray, action: np.ndarray, output_list: Optional[list[str]] = None
    ) -> Optional[Dict]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, env, **kwargs):
        super().__init__(observation_space, action_space, env, **kwargs)

    def act(self, observation: np.ndarray):
        if self.is_vec_env:
            return np.array([self.action_space.sample() for _ in range(observation.shape[0])])
        else:
            return self.action_space.sample()

    def reset(self):
        pass

    def additional_outputs(
        self, observation: np.ndarray, action: np.ndarray, output_list: Optional[list[str]] = None
    ) -> Optional[Dict]:
        return {"log_probs": np.zeros_like(action), "value": np.array([1.0]), "entropy": np.array([1.0])}

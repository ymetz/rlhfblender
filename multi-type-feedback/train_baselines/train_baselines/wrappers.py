import csv
import json
import os
import time
from typing import Any, ClassVar, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np

try:
    from gym3.interop import _vt2space
    from gym3.types import multimap
except ImportError:
    print("Gym3 not loaded - no support for procgen")
from gymnasium import spaces
from gymnasium.core import ObsType
from sb3_contrib.common.wrappers import (
    TimeFeatureWrapper,
)  # noqa: F401 (backward compatibility)
from stable_baselines3.common.type_aliases import GymObs, GymResetReturn, GymStepReturn


class TruncatedOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env: gym.Env, reward_offset: float = 0.0, n_successes: int = 1):
        super().__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.current_successes = 0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        truncated = truncated or self.current_successes >= self.n_successes
        reward = float(reward) + self.reward_offset
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class ActionNoiseWrapper(gym.Wrapper[ObsType, np.ndarray, ObsType, np.ndarray]):
    """
    Add gaussian noise to the action (without telling the agent),
    to test the robustness of the control.

    :param env:
    :param noise_std: Standard deviation of the noise
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action: np.ndarray) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert isinstance(self.action_space, spaces.Box)
        noise = np.random.normal(np.zeros_like(action), np.ones_like(action) * self.noise_std)
        noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return self.env.step(noisy_action)


class ActionSmoothingWrapper(gym.Wrapper):
    """
    Smooth the action using exponential moving average.

    :param env:
    :param smoothing_coef: Smoothing coefficient (0 no smoothing, 1 very smooth)
    """

    def __init__(self, env: gym.Env, smoothing_coef: float = 0.0):
        super().__init__(env)
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None
        # from https://github.com/rail-berkeley/softlearning/issues/3
        # for smoothing latent space
        # self.alpha = self.smoothing_coef
        # self.beta = np.sqrt(1 - self.alpha ** 2) / (1 - self.alpha)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.smoothed_action = None
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        assert self.smoothed_action is not None
        self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
        return self.env.step(self.smoothed_action)


class DelayedRewardWrapper(gym.Wrapper):
    """
    Delay the reward by `delay` steps, it makes the task harder but more realistic.
    The reward is accumulated during those steps.

    :param env:
    :param delay: Number of steps the reward should be delayed.
    """

    def __init__(self, env: gym.Env, delay: int = 10):
        super().__init__(env)
        self.delay = delay
        self.current_step = 0
        self.accumulated_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.current_step = 0
        self.accumulated_reward = 0.0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.accumulated_reward += float(reward)
        self.current_step += 1

        if self.current_step % self.delay == 0 or terminated or truncated:
            reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    Stack past observations and actions to give an history to the agent.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)  # type: ignore[arg-type]

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs, info = self.env.reset(seed=seed)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history(), info

    def step(self, action) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action
        return self._create_obs_from_history(), reward, terminated, truncated, info


class HistoryWrapperObsDict(gym.Wrapper):
    """
    History Wrapper for dict observation.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.observation_space.spaces["observation"], spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space.spaces["observation"]
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces["observation"] = spaces.Box(
            low=low,
            high=high,
            dtype=wrapped_obs_space.dtype,  # type: ignore[arg-type]
        )

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs_dict, info = self.env.reset(seed=seed)
        obs = obs_dict["observation"]
        self.obs_history[..., -obs.shape[-1] :] = obs

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, info

    def step(self, action) -> Tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, Dict]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = obs_dict["observation"]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from https://github.com/LiuWenlin595/FinalProject.

    :param env: Gym environment
    """

    # Supported envs
    velocity_indices: ClassVar[Dict[str, np.ndarray]] = {
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert env.unwrapped.spec is not None
        env_id: str = env.unwrapped.spec.id
        # By default no masking
        self.mask = np.ones_like(env.observation_space.sample())
        try:
            # Mask velocity
            self.mask[self.velocity_indices[env_id]] = 0.0
        except KeyError as e:
            raise NotImplementedError(f"Velocity masking not implemented for {env_id}") from e

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.mask


class Gym3ToGymnasium(gym.Env):
    """
    Create a gym environment from a gym3 environment.

    Notes:
        * The `render()` method does nothing in "human" mode, in "rgb_array" mode the info dict is checked
            for a key named "rgb" and info["rgb"][0] is returned if present
        * `seed()` and `close() are ignored since gym3 environments do not require these methods
        * `reset()` is ignored if used before an episode is complete because gym3 environments
            reset automatically, if `reset()` was called before the end of an episode, a warning is printed

    :param env: gym3 environment to adapt
    """

    def __init__(self, env, render_mode: str = "rgb_array"):
        self.env = env
        assert env.num == 1
        self.observation_space = _vt2space(env.ob_space)
        self.action_space = _vt2space(env.ac_space)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        self.render_mode = render_mode
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    def reset(self, seed=None, options=None):
        _rew, ob, first = self.env.observe()
        if not first[0]:
            print("Warning: early reset ignored")
        return multimap(lambda x: x[0], ob), {}

    def step(self, ac):
        _, prev_ob, _ = self.env.observe()
        self.env.act(np.array([ac]))
        rew, ob, first = self.env.observe()
        if first[0]:
            ob = prev_ob
        return (
            multimap(lambda x: x[0], ob),
            rew[0],
            first[0],
            first[0],
            self.env.get_info()[0],
        )

    def render(self):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if self.render_mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def seed(self, *args):
        print("Warning: seed ignored")

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self


# Metaworld Wrapper
class MetaWorldMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super(MetaWorldMonitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(MetaWorldMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, MetaWorldMonitor.EXT)
                else:
                    filename = filename + "." + MetaWorldMonitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write("#%s\n" % json.dumps({"t_start": self.t_start, "env_id": env.spec and env.spec.id}))
            self.logger = csv.DictWriter(
                self.file_handler,
                fieldnames=("r", "l", "t", "s") + reset_keywords + info_keywords,
            )
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.episode_success = 0
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with MetaWorldMonitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.episode_success = 0
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Expected you to pass kwarg {} into reset".format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        self.episode_success = max(self.episode_success, info["success"])

        if terminated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "s": self.episode_success,
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(MetaWorldMonitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times

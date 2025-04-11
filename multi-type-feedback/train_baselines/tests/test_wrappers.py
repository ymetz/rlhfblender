import gymnasium as gym
import pytest
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import DummyVecEnv

import train_baselines.import_envs  # noqa: F401
from train_baselines.utils import get_wrapper_class
from train_baselines.wrappers import (
    ActionNoiseWrapper,
    DelayedRewardWrapper,
    HistoryWrapper,
    TimeFeatureWrapper,
)


def test_wrappers():
    env = gym.make("Ant-v4")
    env = DelayedRewardWrapper(env)
    env = ActionNoiseWrapper(env)
    env = HistoryWrapper(env)
    env = TimeFeatureWrapper(env)
    check_env(env)


@pytest.mark.parametrize(
    "env_wrapper",
    [
        None,
        {"train_baselines.wrappers.HistoryWrapper": dict(horizon=2)},
        [
            {"train_baselines.wrappers.HistoryWrapper": dict(horizon=3)},
            "train_baselines.wrappers.TimeFeatureWrapper",
        ],
    ],
)
def test_get_wrapper(env_wrapper):
    env = gym.make("Ant-v4")
    hyperparams = {"env_wrapper": env_wrapper}
    wrapper_class = get_wrapper_class(hyperparams)
    if env_wrapper is not None:
        env = wrapper_class(env)
    check_env(env)


@pytest.mark.parametrize(
    "vec_env_wrapper",
    [
        None,
        {"stable_baselines3.common.vec_env.VecFrameStack": dict(n_stack=2)},
        [
            {"stable_baselines3.common.vec_env.VecFrameStack": dict(n_stack=3)},
            "stable_baselines3.common.vec_env.VecMonitor",
        ],
    ],
)
def test_get_vec_env_wrapper(vec_env_wrapper):
    env = gym.make("Ant-v4")
    env = DummyVecEnv([lambda: env])
    hyperparams = {"vec_env_wrapper": vec_env_wrapper}
    wrapper_class = get_wrapper_class(hyperparams, "vec_env_wrapper")
    if wrapper_class is not None:
        env = wrapper_class(env)
    A2C("MlpPolicy", env).learn(16)

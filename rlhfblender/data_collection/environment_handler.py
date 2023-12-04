import importlib
import os
from typing import Union

import gymnasium as gym
from databases import Database
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from rlhfblender.data_handling.database_handler import add_entry
from rlhfblender.data_models.global_models import Environment
from rlhfblender.utils import get_wrapper_class


def get_environment(
    env_name: str = "CartPole-v0",
    n_envs: int = 1,
    environment_config: dict = {},
    norm_env_path: Union[str, None] = None,
    additional_packages: dict = [],
) -> VecEnv:
    """
    Get the gym environment by name.
    Code partially taken from SB Baselines 3
    :param env_name: (str) Name of the environment
    :param n_envs: (int) Number of parallel environments
    :param additional_packages: (dict) Additional packages to import
    :param environment_config: (dict) Environment configuration
    :param norm_env_path: (str) Path to the normalized environment
    :return:
    """
    for env_module in additional_packages:
        importlib.import_module(env_module)

    env_wrapper = get_wrapper_class(environment_config)

    vec_env_cls = DummyVecEnv

    env = make_vec_env(
        env_name,
        n_envs=n_envs,
        wrapper_class=env_wrapper,
        env_kwargs=environment_config.get("env_kwargs", None),
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=environment_config.get("vec_env_kwargs", None),
    )

    if "vec_env_wrapper" in environment_config.keys():
        vec_env_wrapper = get_wrapper_class(environment_config, "vec_env_wrapper")
        env = vec_env_wrapper(env)

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if norm_env_path and environment_config.get("normalize", False):
        print("Loading running average")
        print(f"with params: {environment_config['normalize_kwargs']}")
        path_ = os.path.join(norm_env_path, "vecnormalize.pkl")
        if os.path.exists(path_):
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False
        else:
            raise ValueError(f"VecNormalize stats {path_} not found")

    n_stack = environment_config.get("frame_stack", 0)
    if n_stack > 0:
        print(f"Stacking {n_stack} frames")
        env = VecFrameStack(env, n_stack)

    return env


def initial_space_info(space: gym.spaces.Space) -> dict:
    """
    Get the initial space info for the environment.
    Currently only works for Box, Discrete, MultiDiscrete and MultiBinary spaces.
    :param space:
    :return:
    """
    space_low = None
    space_high = None
    if hasattr(space, "low"):
        space_low = space.low if len(space.shape) < 2 else space.low[0]
    if hasattr(space, "high"):
        space_high = space.high if len(space.shape) < 2 else space.high[0]
    shape = (space.n,) if isinstance(space, gym.spaces.Discrete) else space.shape

    tag_dict = {}
    if shape is not None:
        tag_dict = {f"tag_{i}": i for i in range(shape[-1])}

    return {
        "label": f"{space.__class__.__name__}({shape!s})",
        "shape": shape,
        "low": space_low.tolist() if space_low is not None else [],
        "high": space_high.tolist() if space_high is not None else [],
        "dtype": str(space.dtype),
        **tag_dict,
    }


async def initial_registration(database: Database, env_name: str = "CartPole-v0", additional_gym_packages: list = []) -> None:
    """
    Register the environment with the database.
    :param database: (Database) The database to register the environment with (see database_handler.py)
    :param env_name: (str) The name of the environment
    :return: None
    """
    env = gym.make(env_name, render_mode="rgb_array")

    # Replace inf/-inf values inside of the spaces with the max/min values of the spaces with min/max float values
    if hasattr(env.observation_space, "low"):
        env.observation_space.low[env.observation_space.low == -float("inf")] = -1000
        env.observation_space.low[env.observation_space.low == float("inf")] = 1000
    if hasattr(env.observation_space, "high"):
        env.observation_space.high[env.observation_space.high == -float("inf")] = -1000
        env.observation_space.high[env.observation_space.high == float("inf")] = 1000
    if hasattr(env.action_space, "low"):
        env.action_space.low[env.action_space.low == -float("inf")] = -1000
        env.action_space.low[env.action_space.low == float("inf")] = 1000
    if hasattr(env.action_space, "high"):
        env.action_space.high[env.action_space.high == -float("inf")] = -1000
        env.action_space.high[env.action_space.high == float("inf")] = 1000

    return await add_entry(
        database,
        Environment,
        Environment(
            env_name=env_name,
            registered=1,
            registration_id=env.spec.id,
            observation_space_info=initial_space_info(env.observation_space),
            action_space_info=initial_space_info(env.action_space),
            has_state_loading=0,
            description="",
            tags=[],
            env_path="",
            additional_gym_packages=additional_gym_packages,
        ).dict(),
    )

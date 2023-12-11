import importlib
import os
from typing import Optional, Union

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from rlhfblender.data_models.global_models import Environment
from rlhfblender.utils import get_wrapper_class


def get_environment(
    env_name: str = "CartPole-v0",
    n_envs: int = 1,
    environment_config: Optional[dict] = None,
    norm_env_path: Union[str, None] = None,
    additional_packages: list = (),
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
    if environment_config is None:
        environment_config = {}
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
    Get the initial space info for the environment, in particular the tag dict which is used for the
    the naming of the observation and action space in the user interface.
    :param space:
    :return:
    """
    shape = (space.n,) if isinstance(space, gym.spaces.Discrete) else space.shape

    tag_dict = {}
    if shape is not None:
        tag_dict = {f"tag_{i}": i for i in range(shape[-1])}

    return {
        "label": f"{space.__class__.__name__}({shape!s})",
        "shape": shape,
        "dtype": str(space.dtype),
        **tag_dict,
    }


def initial_registration(
    env_id: str = "CartPole-v0", entry_point: Optional[str] = "", additional_gym_packages: Optional[list] = ()
) -> Environment:
    """
    Register the environment with the database.
    :param database: (Database) The database to register the environment with (see database_handler.py)
    :param env_name: (str) The name of the environment
    :return: None
    """

    if len(additional_gym_packages) > 0:
        for env_module in additional_gym_packages:
            importlib.import_module(env_module)

    if entry_point != "":
        gym.register(id=env_id, entry_point=entry_point)

    env = gym.make(env_id, render_mode="rgb_array")

    return Environment(
        env_name=env_id,
        registered=1,
        registration_id=env_id,
        observation_space_info=initial_space_info(env.observation_space),
        action_space_info=initial_space_info(env.action_space),
        has_state_loading=0,
        description="",
        tags=[],
        env_path="",
        additional_gym_packages=additional_gym_packages,
    )

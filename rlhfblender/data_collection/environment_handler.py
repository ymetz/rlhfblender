import importlib
import os
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper
from train_baselines.utils import get_wrapper_class
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from rlhfblender.data_models.global_models import Environment


def numpy_to_python(obj):
    """
    Converts a NumPy object to native Python types, handling special cases.

    Args:
        obj: Any NumPy object or native Python object

    Returns:
        A native Python object equivalent

    Examples:
        >>> numpy_to_python(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> numpy_to_python(np.inf)
        float('inf')
        >>> numpy_to_python(np.nan)
        float('nan')
    """
    # Handle None
    if obj is None:
        return None

    # Handle NumPy scalars
    if isinstance(obj, np.generic):
        if np.isnan(obj):
            return float("nan")
        elif np.isinf(obj):
            # return float('inf') if obj > 0 else float('-inf')
            # just use very large number instead of float('inf') - not ideal
            return 1e9 if obj > 0 else -1e9
        else:
            return obj.item()

    # Handle NumPy arrays
    elif isinstance(obj, np.ndarray):
        return [numpy_to_python(x) for x in obj]

    # Handle lists and tuples recursively
    elif isinstance(obj, list | tuple):
        return type(obj)(numpy_to_python(x) for x in obj)

    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {numpy_to_python(k): numpy_to_python(v) for k, v in obj.items()}

    # Return other types as-is
    return obj


def get_metaworld_env(
    env_name: str = "pick-place-v3",
    n_envs: int = 1,
    environment_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> VecEnv:
    """
    Create a MetaWorld environment wrapped to match the Gymnasium interface.
    """
    if n_envs != 1:
        raise ValueError("MetaWorld environments currently only support n_envs=1")

    try:
        import metaworld

    except ImportError:
        raise ImportError("Please install MetaWorld to use MetaWorld environments")

    # Initialize MT1 benchmark
    mt1 = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: mt1])

    return vec_env


def get_environment(
    env_name: str = "CartPole-v0",
    n_envs: int = 1,
    environment_config: dict | None = None,
    norm_env_path: str | None = None,
    additional_packages: list = (),
    gym_entry_point: str | None = "",
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

    # Create a wrapper function that applies both the original wrapper and SaveResetEnvWrapper
    def combined_wrapper(env):
        if env_wrapper is not None:
            env = env_wrapper(env)
        # Always wrap with SaveResetEnvWrapper for env_state functionality
        env = SaveResetEnvWrapper(env)
        return env

    vec_env_cls = DummyVecEnv

    print(f"Creating environment {env_name} with config: {environment_config}")
    env_kwargs = environment_config.get("env_kwargs", None)
    # add render_mode = 'rgb_array' to env_kwargs
    if env_kwargs is None:
        env_kwargs = {"render_mode": "rgb_array"}
    else:
        env_kwargs["render_mode"] = "rgb_array"

    if gym_entry_point:
        # Register the environment with the given entry point to gym for the current session
        gym.register(id=env_name, entry_point=gym_entry_point)

    print(f"Creating environment {env_name}")
    if "metaworld" in env_name.lower():
        from train_baselines.utils import make_vec_metaworld_env

        environment_name = env_name.replace("metaworld-", "")
        env = make_vec_metaworld_env(
            env_id=environment_name,
            n_envs=n_envs,
            wrapper_class=combined_wrapper,
            env_kwargs=env_kwargs,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=environment_config.get("vec_env_kwargs", None),
        )
    else:
        env = make_vec_env(
            env_name,
            n_envs=n_envs,
            wrapper_class=combined_wrapper,
            env_kwargs=env_kwargs,
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
    try:
        n_stack = int(n_stack)
    except ValueError:
        print(f"Invalid frame stack value: {n_stack}. Make sure to pass an integer.")
        n_stack = 0
    if n_stack > 0:
        print(f"Stacking {n_stack} frames")
        env = VecFrameStack(env, n_stack)

    return env


def initial_space_info(space: gym.spaces.Space, save_low_high=False, action_names: list[str] | None = None) -> dict:
    """
    Get the initial space info for the environment, in particular the tag dict which is used for the
    the naming of the observation and action space in the user interface.
    :param space:
    :return:
    """
    shape = (space.n,) if isinstance(space, gym.spaces.Discrete) else space.shape

    tag_dict = {}
    if shape is not None:
        if action_names:
            assert len(action_names) == shape[-1], "Action names must match the number of actions"
            tag_dict = {f"{i}": action_names[i] for i in range(shape[-1])}
        else:
            tag_dict = {f"{i}": i for i in range(shape[-1])}

    return_dict = {
        "label": f"{space.__class__.__name__}({shape!s})",
        "shape": shape,
        "dtype": str(space.dtype),
        "labels": tag_dict,
    }

    if save_low_high:
        if hasattr(space, "low"):
            return_dict["low"] = numpy_to_python(space.low)
        if hasattr(space, "high"):
            return_dict["high"] = numpy_to_python(space.high)

    return return_dict


def initial_registration(
    env_id: str = "CartPole-v0",
    entry_point: str | None = "",
    additional_gym_packages: list | None = (),
    gym_env_kwargs: dict | None = None,
    action_names: list[str] | None = None,
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

    # Check if this is a MetaWorld environment
    is_metaworld = "metaworld" in env_id.lower()
    environment_id = env_id.replace("metaworld-", "")

    if is_metaworld:
        env = get_metaworld_env(environment_id).envs[0]
    else:
        if entry_point:
            gym.register(id=env_id, entry_point=entry_point)
        env = gym.make(env_id, render_mode="rgb_array", **gym_env_kwargs)

    return Environment(
        env_name=env_id,
        registered=1,
        registration_id=env_id,
        observation_space_info=initial_space_info(env.observation_space),
        action_space_info=initial_space_info(env.action_space, save_low_high=True, action_names=action_names),
        has_state_loading=0,
        description="",
        tags=[],
        gym_entry_point=entry_point if entry_point else "",
        additional_gym_packages=[] if len(additional_gym_packages) == 0 else additional_gym_packages,
    )

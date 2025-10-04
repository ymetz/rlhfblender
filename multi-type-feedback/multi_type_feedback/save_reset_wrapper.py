import copy
import importlib
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


def _has_attrs(obj, *attrs) -> bool:
    return all(hasattr(obj, a) for a in attrs)


def _is_mujoco(env) -> bool:
    """
    Duck-type check for Gymnasium MuJoCo envs:
    - .data with qpos/qvel
    - .set_state(qpos, qvel)
    """
    u = env.unwrapped
    return (
        hasattr(u, "data")
        and _has_attrs(u.data, "qpos", "qvel")
        and hasattr(u, "set_state")
    )


def _is_atari(env) -> bool:
    """
    ALE-based envs usually expose clone_state/restore_state.
    """
    u = env.unwrapped
    if _has_attrs(u, "clone_state", "restore_state"):
        return True
    # Fallback: module name hint (doesn't import ale_py)
    mod = getattr(u.__class__, "__module__", "")
    return "ale_py" in mod or "atari" in mod.lower()


def _is_minigrid(env) -> bool:
    """
    Minigrid envs expose these attributes directly.
    """
    u = env.unwrapped
    needed = ("grid", "carrying", "agent_pos", "agent_dir", "mission")
    if _has_attrs(u, *needed):
        return True
    mod = getattr(u.__class__, "__module__", "")
    return "minigrid" in mod.lower()


def _load_custom_wrappers():
    """
    Load your custom wrapper registry lazily to avoid importing it
    unless needed. If it doesn't exist, return empty mapping.
    """
    try:
        mod = importlib.import_module("multi_type_feedback.custom_save_reset_envs")
        return getattr(mod, "CUSTOM_SAVE_RESET_WRAPPERS", {})
    except Exception:
        return {}


class SaveResetEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        A Wrapper that ensures ability to save/load the state for a selected
        set of environments without hard dependencies. If you encounter an
        unsupported env, add a custom wrapper in your lazy registry.
        """
        super().__init__(env)

        # Try to match and attach a custom wrapper (lazily imported)
        env_id = env.spec.id if getattr(env, "spec", None) is not None else env.__class__.__name__
        custom = _load_custom_wrappers()
        for key, wrapper_ctor in custom.items():
            if key in env_id:
                # Replace the unwrapped env with the custom save/load wrapper.
                self.env = wrapper_ctor(self.env)
                break

    # --------- Save / Load ---------

    def save_state(self, observation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Save the current state of the environment and return it.
        This avoids importing any optional env packages by using duck-typing.
        """
        u = self.unwrapped

        if _is_mujoco(self):
            # MuJoCo (duck-typed)
            state = {
                "qpos": np.copy(u.data.qpos),
                "qvel": np.copy(u.data.qvel),
            }
        elif _is_atari(self):
            # ALE/Atari
            state = u.clone_state()
        elif _is_minigrid(self):
            # MiniGrid (assignable attributes)
            state = {
                "grid": u.grid,
                "carrying": u.carrying,
                "agent_pos": u.agent_pos,
                "agent_dir": u.agent_dir,
                "mission": u.mission,
            }
        elif hasattr(self.env, "get_state"):
            # Generic get_state API (incl. Gym3ToGymnasium inner envs)
            state = self.env.get_state()
        elif hasattr(u, "get_state"):
            state = u.get_state()
        else:
            # Last resort: deepcopy the whole unwrapped env (can be heavy)
            state = copy.deepcopy(u)

        # Preserve elapsed steps for TimeLimit wrapper semantics
        elapsed_steps = getattr(self, "_elapsed_steps", 0)

        return {
            "state": state,
            "observation": observation,
            "elapsed_steps": elapsed_steps,
        }

    def load_state(self, state_and_obs: Dict[str, Any]):
        """
        Load a given state into the environment via duck-typing.
        Returns the observation (either stored or recomputed by custom wrappers).
        """
        if state_and_obs is None:
            raise ValueError("The provided state is None. Please provide a valid state.")

        obs = state_and_obs.get("observation", None)
        state = state_and_obs["state"]
        self._elapsed_steps = state_and_obs.get("elapsed_steps", 0)

        u = self.unwrapped

        if _is_mujoco(self):
            # Accept both "qvel" and legacy "qval" for backward compatibility
            qpos = state["qpos"]
            qvel = state.get("qvel", state.get("qval"))
            if qvel is None:
                raise KeyError("MuJoCo state missing 'qvel'/'qval'.")
            u.set_state(qpos, qvel)
        elif _is_atari(self):
            u.restore_state(state)
        elif _is_minigrid(self):
            u.grid = state["grid"]
            u.carrying = state["carrying"]
            u.agent_pos = state["agent_pos"]
            u.agent_dir = state["agent_dir"]
            u.mission = state["mission"]
        elif hasattr(self.env, "set_state"):
            self.env.set_state(state)
        elif hasattr(u, "set_state"):
            u.set_state(state)
        else:
            # Blind attribute copy (best effort)
            for attr, value in vars(state).items():
                setattr(u, attr, value)

        return obs

    def reset(self, **kwargs):
        """Normal reset; state loading is explicit via load_state()."""
        observation, info = super().reset(**kwargs)
        return observation, info
import copy

import gymnasium as gym
import numpy as np
from ale_py import AtariEnv
from gymnasium.envs.mujoco import MujocoEnv
from minigrid.minigrid_env import MiniGridEnv

from multi_type_feedback.custom_save_reset_envs import CUSTOM_SAVE_RESET_WRAPPERS
from train_baselines.wrappers import Gym3ToGymnasium


class SaveResetEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        A Wrapper that ensure ability to save/load the state for a selected
        set of enviornments. If you encounter an unsupported env, feel free to
        add them to the CUSTOM_SAVE_RESET_WRAPPERS dict
        """
        super(SaveResetEnvWrapper, self).__init__(env)

        env_id = env.spec.id if env.spec is not None else env.__class__.__name__
        for key, wrapper_ctor in CUSTOM_SAVE_RESET_WRAPPERS.items():
            if key in env_id:
                # Replace the unwrapped env with the custom save/load wrapper.
                self.env = wrapper_ctor(self.env)
                break

    def save_state(self, observation=None):
        """
        Save the current state of the environment and return it.
        For MuJoCo environments, use the env.sim.get_state() method.
        For MiniGrid environments, use copy.deepcopy() to save the state.
        """
        if isinstance(self.unwrapped, MujocoEnv):
            # MuJoCo environment
            state = {
                "qpos": np.copy(self.unwrapped.data.qpos),
                "qval": np.copy(self.unwrapped.data.qvel),
            }
        elif isinstance(self.unwrapped, AtariEnv):
            state = self.unwrapped.clone_state()
        elif isinstance(self.unwrapped, MiniGridEnv):
            # Minigrid environment
            state = {
                "grid": self.unwrapped.grid,
                "carrying": self.unwrapped.carrying,
                "agent_pos": self.unwrapped.agent_pos,
                "agent_dir": self.unwrapped.agent_dir,
                "mission": self.unwrapped.mission,
            }
        elif isinstance(self.unwrapped, Gym3ToGymnasium):
            state = self.unwrapped.env.get_state()
        elif hasattr(self.env, "get_state"):
            state = self.env.get_state()
        else:
            # Something else
            state = copy.deepcopy(self.unwrapped)

        # YM: For TimeLimit Gymnasium Wrapper, also set the elapsed timesteps to original value
        # to avoid late termination (in genenerate_feedback.py the el. timesteps are always set to 0)
        if hasattr(self, "_elapsed_steps"):
            elapsed_steps = self._elapsed_steps
        else:
            elapsed_steps = 0

        return {
            "state": state,
            "observation": observation,
            "elapsed_steps": elapsed_steps,
        }

    def load_state(self, state_and_obs):
        """
        Load a given state into the environment.
        For MuJoCo environments, use the env.sim.set_state() method.
        For MiniGrid environments, directly assign the provided state.
        Returns the observation
        """
        if state_and_obs is None:
            raise ValueError(
                "The provided state is None. Please provide a valid state."
            )

        obs = state_and_obs["observation"]
        state = state_and_obs["state"]
        self._elapsed_steps = state_and_obs.get("elapsed_steps", 0)

        if isinstance(self.unwrapped, MujocoEnv):
            # MuJoCo environment
            self.unwrapped.set_state(state["qpos"], state["qval"])
        elif isinstance(self.unwrapped, AtariEnv):
            self.unwrapped.restore_state(state)
        elif isinstance(self.unwrapped, MiniGridEnv):
            # Minigrid environment (A bit cluncky i guess)
            self.unwrapped.grid = state["grid"]
            self.unwrapped.carrying = state["carrying"]
            self.unwrapped.agent_pos = state["agent_pos"]
            self.unwrapped.agent_dir = state["agent_dir"]
            self.unwrapped.mission = state["mission"]
        elif isinstance(self.unwrapped, Gym3ToGymnasium):
            self.unwrapped.env.set_state(state)
        elif hasattr(self.env, "set_state"):
            self.env.set_state(state)
        else:
            # Other environments
            for attr, value in vars(state).items():
                setattr(self.unwrapped, attr, value)
        return obs

    def reset(self, **kwargs):
        """
        Reset the environment. If a state is provided, load it after resetting.
        Otherwise, perform a normal reset.
        """
        observation, info = super(SaveResetEnvWrapper, self).reset(**kwargs)
        return observation, info

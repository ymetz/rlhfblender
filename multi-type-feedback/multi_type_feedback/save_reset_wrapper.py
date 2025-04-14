import copy

import gymnasium as gym
import numpy as np

try:
    from ale_py import AtariEnv
except ImportError:
    AtariEnv = None
from gymnasium.envs.mujoco import MujocoEnv
from minigrid.minigrid_env import MiniGridEnv
from train_baselines.wrappers import Gym3ToGymnasium

from multi_type_feedback.custom_save_reset_envs import CUSTOM_SAVE_RESET_WRAPPERS


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
        elif AtariEnv is not None and isinstance(self.unwrapped, AtariEnv):
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

        return {"state": state, "observation": observation, "elapsed_steps": elapsed_steps}

    def load_state(self, state_and_obs):
        """
        Load a given state into the environment.
        For MuJoCo environments, use the env.sim.set_state() method.
        For MiniGrid environments, directly assign the provided state.
        Returns the observation
        """
        if state_and_obs is None:
            raise ValueError("The provided state is None. Please provide a valid state.")

        obs = state_and_obs["observation"]
        state = state_and_obs["state"]
        self._elapsed_steps = state_and_obs.get("elapsed_steps", 0)

        if isinstance(self.unwrapped, MujocoEnv):
            # MuJoCo environment
            self.unwrapped.set_state(state["qpos"], state["qval"])
        elif AtariEnv is not None and isinstance(self.unwrapped, AtariEnv):
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

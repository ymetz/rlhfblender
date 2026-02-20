import gymnasium as gym
import numpy as np

"""
    For a list of supported environments, see the CUSTOM_ENVS dict at the bottom of the file
"""

class LunarLanderSaveLoadWrapper(gym.Wrapper):
    """
    A wrapper that adds save_state and load_state methods to a LunarLander environment.
    It captures the dynamic state of the lander and its legs, and restores them
    by first calling reset() (to reinitialize the Box2D world) and then setting the state.
    """

    def __init__(self, env):
        super().__init__(env)
        # Dynamically import constants (Box2D-related) only when needed.
        from gymnasium.envs.box2d.lunar_lander import (
            FPS,
            LEG_DOWN,
            SCALE,
            VIEWPORT_H,
            VIEWPORT_W,
        )

        self.VIEWPORT_W = VIEWPORT_W
        self.VIEWPORT_H = VIEWPORT_H
        self.SCALE = SCALE
        self.FPS = FPS
        self.LEG_DOWN = LEG_DOWN

        print("DOING THE WRAP")

    def get_state(self):
        """
        Save the essential state of the LunarLander.
        Returns a dictionary containing the state and (optionally) an observation.
        """
        if self.unwrapped.lander is None:
            raise ValueError("Lander does not exist. Cannot save state.")
        state = {
            "lander": {
                "position": (
                    self.unwrapped.lander.position.x,
                    self.unwrapped.lander.position.y,
                ),
                "angle": self.unwrapped.lander.angle,
                "linearVelocity": (
                    self.unwrapped.lander.linearVelocity.x,
                    self.unwrapped.lander.linearVelocity.y,
                ),
                "angularVelocity": self.unwrapped.lander.angularVelocity,
            },
            "legs": [
                {
                    "position": (leg.position.x, leg.position.y),
                    "angle": leg.angle,
                    "ground_contact": leg.ground_contact,
                }
                for leg in self.unwrapped.legs
            ],
            # Save wind indices if wind is enabled.
            "wind_idx": self.unwrapped.wind_idx if self.unwrapped.enable_wind else None,
            "torque_idx": (
                self.unwrapped.torque_idx if self.unwrapped.enable_wind else None
            ),
            "prev_shaping": self.unwrapped.prev_shaping,
        }
        return state

    def set_state(self, state):
        """
        Load a previously saved state into the LunarLander environment.
        First calls reset() to reinitialize the Box2D world, then restores the state.
        Returns an observation computed from the restored state.
        """
        if state is None:
            raise ValueError("No state provided for loading.")

        # Restore lander state.
        lander_state = state["lander"]
        self.unwrapped.lander.position = lander_state["position"]
        self.unwrapped.lander.angle = lander_state["angle"]
        self.unwrapped.lander.linearVelocity = lander_state["linearVelocity"]
        self.unwrapped.lander.angularVelocity = lander_state["angularVelocity"]

        # Restore legs state.
        legs_state = state["legs"]
        if len(legs_state) != len(self.unwrapped.legs):
            raise ValueError("Mismatch in number of legs in saved state.")
        for leg, saved_leg in zip(self.unwrapped.legs, legs_state):
            leg.position = saved_leg["position"]
            leg.angle = saved_leg["angle"]
            leg.ground_contact = saved_leg["ground_contact"]

        if self.unwrapped.enable_wind:
            self.unwrapped.wind_idx = state.get("wind_idx", self.unwrapped.wind_idx)
            self.unwrapped.torque_idx = state.get(
                "torque_idx", self.unwrapped.torque_idx
            )
        self.unwrapped.prev_shaping = state.get("prev_shaping", None)

        return self.get_obs()

    def get_obs(self):
        """
        Recompute and return the observation based on the current state.
        This replicates the computation in LunarLander.step.
        """
        pos = self.unwrapped.lander.position
        vel = self.unwrapped.lander.linearVelocity
        obs = np.array(
            [
                (pos.x - self.VIEWPORT_W / self.SCALE / 2)
                / (self.VIEWPORT_W / self.SCALE / 2),
                (pos.y - (self.unwrapped.helipad_y + self.LEG_DOWN / self.SCALE))
                / (self.VIEWPORT_H / self.SCALE / 2),
                vel.x * (self.VIEWPORT_W / self.SCALE / 2) / self.FPS,
                vel.y * (self.VIEWPORT_H / self.SCALE / 2) / self.FPS,
                self.unwrapped.lander.angle,
                20.0 * self.unwrapped.lander.angularVelocity / self.FPS,
                1.0 if self.unwrapped.legs[0].ground_contact else 0.0,
                1.0 if self.unwrapped.legs[1].ground_contact else 0.0,
            ],
            dtype=np.float32,
        )
        return obs


CUSTOM_SAVE_RESET_WRAPPERS = {
    "LunarLander": lambda env: LunarLanderSaveLoadWrapper(env),
}
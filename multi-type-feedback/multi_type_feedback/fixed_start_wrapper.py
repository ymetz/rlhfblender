"""FixedStartWrapper: gymnasium wrapper that resets every episode to the same initial state.

Design
------
Extends SaveResetEnvWrapper so it inherits the env-type-aware save/load logic
(MuJoCo qpos/qvel, Atari clone_state, MiniGrid attributes, custom wrappers, …).

Sharing state across runs (baseline training + RLHF training)
-------------------------------------------------------------
Use ``sample_fixed_state`` with a *cache_path*. The first call samples and saves
the state; every subsequent call loads from the file. Point all scripts at the
same path (or use the same ``--state-seed``) and they will all use identical
physics configurations.

    fixed_state = sample_fixed_state(
        env_name, state_seed,
        cache_path="configs/fixed_states/sweep_into_v3_seed0.pkl",
    )

Usage — synchronized VecEnv (all workers start from identical state)
----------------------------------------------------------------------
    exp_manager.env_wrapper = lambda env: FixedStartWrapper(env, fixed_state=fixed_state)

Usage — single env (state locked after first reset)
----------------------------------------------------
    env = FixedStartWrapper(gym.make("Pendulum-v1"))
    obs, _ = env.reset()  # state is sampled & locked here
    obs, _ = env.reset()  # restores the same state
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym

from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper


class FixedStartWrapper(SaveResetEnvWrapper):
    """Reset every episode to the same fixed initial state.

    Parameters
    ----------
    env:
        The (possibly already-wrapped) gymnasium environment.
    fixed_state:
        A state dict as returned by ``SaveResetEnvWrapper.save_state()``.
        If *None*, the state is sampled on the very first ``reset()`` call
        and locked for all subsequent calls.
        Pass the **same** dict to every worker env so they all start identically.
    """

    def __init__(self, env: gym.Env, fixed_state: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        self._fixed_state = fixed_state

    def reset(self, **kwargs):
        # Always perform a real env reset first so that
        # - TimeLimit counters are reset
        # - internal env structures are re-initialized (important for Metaworld)
        obs, info = self.env.reset(**kwargs)

        if self._fixed_state is None:
            # First call: capture and lock the current state.
            self._fixed_state = self.save_state(observation=obs)
        else:
            # Subsequent calls: restore the fixed physics state.
            stored_obs = self.load_state(self._fixed_state)
            if stored_obs is not None:
                obs = stored_obs

        return obs, info


def _default_cache_path(env_name: str, seed: int) -> Path:
    # Anchor to the repo root (two levels above this file:
    #   fixed_start_wrapper.py → multi_type_feedback/ → multi-type-feedback/ → repo root)
    # so the path is the same regardless of which directory the script is run from.
    repo_root = Path(__file__).resolve().parents[2]
    safe = env_name.replace("/", "_").replace("-", "_").replace(" ", "_")
    return repo_root / "configs" / "fixed_states" / f"{safe}_seed{seed}.pkl"


def sample_fixed_state(
    env_name: str,
    seed: int,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a fixed initial state for *env_name*, sampling if needed.

    Parameters
    ----------
    env_name:
        Gymnasium environment ID (e.g. ``"metaworld-sweep-into-v3"``).
    seed:
        Seed used both for ``gym.make`` and to name the cache file.
        **Use the same seed across baseline training and RLHF training** so
        that all runs share the exact same initial physics configuration.
    cache_path:
        Path to a ``.pkl`` file.
        - If the file already exists it is loaded directly (no env is created).
        - If it does not exist the state is sampled and saved there.
        Defaults to ``configs/fixed_states/<env>_seed<seed>.pkl`` when *None*.

    Returns
    -------
    dict
        State dict compatible with ``SaveResetEnvWrapper.load_state()``.
    """
    if cache_path is None:
        cache_path = str(_default_cache_path(env_name, seed))

    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"  Loading fixed start state from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # Sample from a temporary env.
    from multi_type_feedback.utils import TrainingUtils

    tmp = TrainingUtils.setup_environment(env_name, seed)
    obs, _ = tmp.reset()
    state = tmp.save_state(observation=obs)
    tmp.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(state, fh)
    print(f"  Fixed start state saved to: {cache_path}")

    return state

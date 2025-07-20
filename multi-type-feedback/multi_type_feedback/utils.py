import abc
import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import wandb
from gymnasium.wrappers import FrameStackObservation, TransformObservation
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecEnvWrapper,
    VecNormalize,
)

from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper
from train_baselines.wrappers import Gym3ToGymnasium

try:
    import minigrid
except ImportError:
    print("Cannot import minigrid")

try:
    import highway_env
except ImportError:
    print("Cannot import highway env")
try:
    import metaworld
except ImportError:
    print("Cannot import metaworld")
try:
    from procgen import ProcgenGym3Env

    from train_baselines.wrappers import Gym3ToGymnasium
except ImportError:
    print("Cannot import procgen")


# for convenice sake, todo: make dynamic in the future
discount_factors = {
    "HalfCheetah-v5": 0.98,
    "Hopper-v5": 0.99,
    "Swimmer-v5": 0.9999,
    "Ant-v5": 0.99,
    "Walker2d-v5": 0.99,
    "ALE/BeamRider-v5": 0.99,
    "ALE/MsPacman-v5": 0.99,
    "ALE/Enduro-v5": 0.99,
    "ALE/Pong-v5": 0.99,
    "Humanoid-v5": 0.99,
    "highway-fast-v0": 0.8,
    "merge-v0": 0.8,
    "roundabout-v0": 0.8,
    "metaworld-sweep-into-v2": 0.99,
    "metaworld-button-press-v2": 0.99,
    "metaworld-pick-place-v2": 0.99,
}

class TrainingUtils:
    @staticmethod
    def setup_environment(
        env_name: str, seed: Optional[int] = None, env_kwargs: Optional[Dict[str, Any]] = None, save_reset_wrapper: bool = True
    ) -> gym.Env | SaveResetEnvWrapper:
        """Create and configure the environment based on the environment name."""
        if "procgen" in env_name:
            _, short_name, _ = env_name.split("-")
            environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
            environment = TransformObservation(
                environment, lambda obs: obs["rgb"], environment.observation_space
            )
        elif "ALE/" in env_name:
            environment = FrameStackObservation(AtariWrapper(gym.make(env_name, **(env_kwargs or {}))), 4)
            environment = TransformObservation(
                environment, lambda obs: obs.squeeze(-1), environment.observation_space
            )
        elif "MiniGrid" in env_name:
            environment = FlatObsWrapper(gym.make(env_name, **(env_kwargs or {})))
        elif "metaworld" in env_name:
            environment_name = env_name.replace("metaworld-", "")
            environment = gym.make(
                "Meta-World/MT1",
                env_name=environment_name,
                seed=seed if seed is not None else random.randint(0, 10000),
                **(env_kwargs or {}),
            )
        else:
            environment = gym.make(env_name, **(env_kwargs or {}))

        if save_reset_wrapper:
            environment = SaveResetEnvWrapper(environment)

        return environment

    @staticmethod
    def setup_base_parser() -> argparse.ArgumentParser:
        """Create a base argument parser with common arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--environment", type=str, default="HalfCheetah-v5", help="Environment name"
        )
        parser.add_argument(
            "--environment-kwargs",
            type=str,
            nargs="+",
            help='Environment Kwargs (e.g. --env-kwargs key1:value1 key2:value2), e.g.: \
                "env_wrapper:stable_baselines3.common.atari_wrappers.AtariWrapper frame_stack:4"',
            default=[],
        )
        parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
        parser.add_argument("--seed", type=int, default=12, help="Random seed")
        parser.add_argument(
            "--n-feedback", type=int, default=-1, help="Number of feedback instances"
        )
        parser.add_argument(
            "--noise-level",
            type=float,
            default=0.0,
            help="Noise level to add to feedback/demonstrations",
        )
        parser.add_argument(
            "--wandb-project-name", default="dynamic_rlhf", help="W&B project name"
        )
        return parser

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def get_device() -> str:
        """Get the appropriate device (CPU/CUDA)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_model_ids(args: argparse.Namespace) -> Tuple[str, str]:
        """Generate feedback and model IDs based on arguments."""
        env_name = (
            args.environment
            if "ALE" not in args.environment
            else args.environment.replace("/", "-")
        )
        feedback_id = f"{args.algorithm}_{env_name}_{args.seed}"
        model_id = (
            f"{feedback_id}_{getattr(args, 'feedback_type', 'default')}_{args.seed}"
        )

        if args.noise_level > 0.0:
            model_id = f"{model_id}_noise_{str(args.noise_level)}"
        if args.n_feedback != -1:
            model_id = f"{model_id}_nfeedback_{str(args.n_feedback)}"

        return feedback_id, model_id

    @staticmethod
    def load_expert_models(
        env_name: str,
        algorithm: str,
        checkpoints_path: str,
        environment: gym.Env,
        top_n_models: Optional[int] = None,
    ) -> list:
        """Load expert models for the given environment."""
        expert_model_paths = [
            os.path.join(checkpoints_path, algorithm, model)
            for model in os.listdir(os.path.join(checkpoints_path, algorithm))
            if env_name in model
        ]

        if top_n_models:
            try:
                run_eval_scores = pd.read_csv(
                    os.path.join(checkpoints_path, "collected_results.csv")
                )
                run_eval_scores = (
                    run_eval_scores.loc[run_eval_scores["env"] == env_name]
                    .sort_values(by=["eval_score"], ascending=False)
                    .head(top_n_models)["run"]
                    .to_list()
                )
                expert_model_paths = [
                    path
                    for path in expert_model_paths
                    if path.split(os.path.sep)[-1] in run_eval_scores
                ]
            except:
                print("[WARN] No eval benchmark results available.")

        expert_models = []
        model_class = PPO if algorithm == "ppo" else SAC

        for expert_model_path in expert_model_paths:
            if os.path.isfile(
                os.path.join(expert_model_path, env_name, "vecnormalize.pkl")
            ):
                norm_env = VecNormalize.load(
                    os.path.join(expert_model_path, env_name, "vecnormalize.pkl"),
                    DummyVecEnv([lambda: environment]),
                )
            else:
                norm_env = None

            model_path = os.path.join(
                expert_model_path,
                f"{env_name}.zip" if "ALE" not in env_name else "best_model.zip",
            )
            expert_models.append((model_class.load(model_path), norm_env))

        return expert_models


    @staticmethod  
    def parse_env_kwargs(env_kwargs: list[str]) -> Dict[str, Any]:
        """Parse environment keyword arguments from a list of strings."""
        env_kwargs_dict = {}
        for kwarg in env_kwargs:
            try:
                key, value = kwarg.split(":")
                env_kwargs_dict[key] = value
            except ValueError:
                print(f"Invalid env_kwargs format: {kwarg}. Expected format is key:value.")
                raise
        return env_kwargs_dict

    @staticmethod
    def setup_wandb_logging(
        model_id: str,
        args: argparse.Namespace,
        additional_config: Optional[Dict[str, Any]] = None,
        wandb_project_name: str = "multi-type-feedback",
    ) -> Any:
        """Initialize W&B logging with given configuration."""
        config = {
            **vars(args),
            "seed": args.seed,
            "environment": args.environment,
            "n_feedback": args.n_feedback,
        }

        if additional_config:
            config.update(additional_config)

        return wandb.init(
            name=model_id,
            project=wandb_project_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=False,
        )


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "multi_type_feedback":
        return current_file.parent.parent
    return current_file.parent


class RewardVecEnvWrapper(VecEnvWrapper):
    """
    A vectorized environment wrapper that modifies the reward
    using a user-defined reward function `reward_fn`.

    :param venv: The vectorized environment to wrap.
    :param reward_fn: A callable that takes in (observations, actions)
                      and returns a modified reward vector.
    """

    def __init__(self, venv: VecEnv, reward_fn):
        super().__init__(venv)
        self.reward_fn = reward_fn
        self.last_actions = None

    def step_async(self, actions: np.ndarray) -> None:
        """
        Forward the actions to the underlying environment asynchronously
        but store them so that we can modify the rewards in step_wait.
        """
        self.last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        """
        Wait for the environment to complete the step, then fetch the data:
        observations, rewards, dones, infos.
        We then apply the custom reward function to generate new rewards
        based on the last actions and the resulting observations.
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        new_rewards = self.reward_fn(obs, self.last_actions)

        return obs, new_rewards, dones, infos

    def reset(self):
        """
        Reset the environment and return initial observations.
        """
        return self.venv.reset()


class L2RegulationCallback(pytorch_lightning.Callback):
    """Callback to dynamically adjust L2 regularization to keep validation loss in target range."""

    def __init__(self, initial_l2=0.01, min_ratio=1.1, max_ratio=1.5):
        super().__init__()
        self.initial_l2 = initial_l2
        self.min_ratio = min_ratio  # Minimum acceptable val/train loss ratio
        self.max_ratio = max_ratio  # Maximum acceptable val/train loss ratio
        self.current_l2 = initial_l2
        self.train_loss = None

    def on_train_epoch_start(self, trainer, pl_module):
        # Apply current L2 regularization
        for param_group in trainer.optimizers[0].param_groups:
            param_group["weight_decay"] = self.current_l2

    def on_train_epoch_end(self, trainer, pl_module):
        # Store training loss
        self.train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip adjustment if missing losses
        if self.train_loss is None or "val_loss" not in trainer.callback_metrics:
            return

        val_loss = float(trainer.callback_metrics["val_loss"])
        loss_ratio = val_loss / self.train_loss

        # Adjust L2 regularization based on validation/training loss ratio
        if loss_ratio < self.min_ratio:
            # Increase regularization if validation is too far from training
            self.current_l2 = min(1.0, self.current_l2 * 1.2)
        elif loss_ratio > self.max_ratio:
            # Reduce regularization if validation is too close to training
            self.current_l2 = max(1e-6, self.current_l2 * 0.8)

        # Log current L2 value
        trainer.logger.log_metrics({"l2_regularization": self.current_l2})


class RewardFn(Protocol):
    """Abstract class for reward function.

    Requires implementation of __call__() to compute the reward given a batch of
    states, actions, next states and dones.
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,`).
        """  # noqa: DAR202
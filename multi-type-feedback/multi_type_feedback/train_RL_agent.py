"""Module for training an RL agent."""

import os
import typing

import gymnasium as gym

# register custom envs
import numpy
import torch

from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
)
from multi_type_feedback.utils import RewardFn, TrainingUtils
from train_baselines.exp_manager import ExperimentManager


class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(
        self,
        reward_model_cls: typing.Union[SingleNetwork, SingleCnnNetwork] = None,
        reward_model_path: list[str] = [],
        vec_env_norm_fn: typing.Optional[typing.Callable] = None,
        action_is_discrete: bool = False,
        action_dim: int = 1,
        device: str = "cuda",
    ):
        """Initialize custom reward."""
        super().__init__()
        self.device = device

        self.reward_model = reward_model_cls.load_from_checkpoint(reward_model_path, map_location=device)

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0
        self.action_is_discrete = action_is_discrete
        self.n_discrete_actions = action_dim

    def _one_hot_encode_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert nested batch of discrete actions to one-hot encoded format.

        Args:
            actions: Tensor of shape (1, batch_size) containing discrete action indices

        Returns:
            one_hot_actions: Tensor of shape (1, batch_size, n_discrete_actions)
        """
        outer_batch, inner_batch = actions.shape
        one_hot = torch.zeros((outer_batch, inner_batch, self.n_discrete_actions), device=self.device)
        actions = actions.long().unsqueeze(-1)  # Add dimension for scatter
        return one_hot.scatter_(2, actions, 1)

    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""

        state = torch.as_tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float).unsqueeze(0)

        if self.action_is_discrete:
            actions = self._one_hot_encode_batch(actions)

        with torch.no_grad():
            if self.reward_model.ensemble_count > 1:
                state = state.expand(self.reward_model.ensemble_count, *state.shape[1:])
                actions = actions.expand(self.reward_model.ensemble_count, *actions.shape[1:])

            rewards = self.reward_model(
                state,
                actions,
            )
            # Reshape rewards to always have 3 dimensions: (ensemble_count, batch_size, 1)
            rewards = rewards.view(self.reward_model.ensemble_count, -1, 1)
            # Take mean across ensemble dimension (dim=0)
            mean_rewards = torch.mean(rewards, dim=0).squeeze(-1)

            return mean_rewards.cpu().numpy()


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--reward-model-folder",
        type=str,
        default="reward_models",
        help="Folder of trained reward models",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="trained_agents",
        help="Folder for finished feedback RL agents",
    )
    parser.add_argument("--feedback-type", type=str, default="evaluative", help="Type of feedback")
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    _, model_id = TrainingUtils.get_model_ids(args)
    reward_model_path = (
        os.path.join(args.reward_model_folder, f"{model_id}.ckpt") if args.feedback_type != "baseline" else None
    )

    TrainingUtils.setup_wandb_logging(f"RL_{model_id}", args, wandb_project_name=args.wandb_project_name)

    architecture_cls = SingleCnnNetwork if "ALE/" in args.environment or "procgen" in args.environment else SingleNetwork

    # we initialize just for the action space, there should be a more elegant way
    # to initialize the CustomRewardFn in the Exp. Manager
    action_space = gym.make(args.environment).action_space
    action_is_discrete = isinstance(action_space, gym.spaces.Discrete)
    action_dim = numpy.prod(action_space.shape) if not action_is_discrete else action_space.n

    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join("agents", f"RL_{model_id}"),
        tensorboard_log=f"runs/RL_{model_id}",
        seed=args.seed,
        log_interval=-1,
        reward_function=(
            CustomReward(
                reward_model_cls=architecture_cls,
                reward_model_path=reward_model_path,
                action_is_discrete=action_is_discrete,
                action_dim=action_dim,
                device=TrainingUtils.get_device(),
            )
            if args.feedback_type != "baseline"
            else None
        ),
        use_wandb_callback=True,
    )

    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()

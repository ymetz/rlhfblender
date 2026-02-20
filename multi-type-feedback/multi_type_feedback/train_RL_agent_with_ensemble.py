"""Module for training an RL agent with weighted ensemble of reward functions."""

import os
import typing

import numpy
import pytorch_lightning as pl
import torch
import wandb
from imitation.rewards.reward_function import RewardFn
from stable_baselines3.common.utils import set_random_seed
from train_baselines.exp_manager import ExperimentManager

from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
)
from multi_type_feedback.utils import TrainingUtils


class CustomReward(RewardFn):
    """Custom reward based on weighted ensemble of fine-tuned reward models."""

    def __init__(
        self,
        reward_model_cls: typing.Union[SingleNetwork, SingleCnnNetwork] = None,
        reward_model_paths: list[str] = [],
        vec_env_norm_fn: typing.Callable = None,
        device: str = "cuda",
        inverse_scaling: bool = False,
    ):
        """Initialize custom reward."""
        super().__init__()
        self.device = device

        self.reward_models = [reward_model_cls.load_from_checkpoint(path, map_location=device) for path in reward_model_paths]

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0

        # Variables for calculating a running mean
        self.reward_mean = None
        self.squared_distance_from_mean = None

        self.inverse_scaling = inverse_scaling

    def standardize_rewards(self, rewards: torch.Tensor):
        """
        Standardizes the input using the rolling mean and standard deviation of the rewards.

        Input should be a tensor of shape (batch_size, model_count).
        """
        model_count = len(self.reward_models)

        if self.reward_mean is None:
            self.reward_mean = torch.zeros(model_count).to(self.device)

        if self.squared_distance_from_mean is None:
            self.squared_distance_from_mean = torch.zeros(model_count).to(self.device)

        standard_deviation = torch.ones(model_count).to(self.device)

        for reward_index, reward in enumerate(rewards):
            # Welford's algorithm for calculating running mean and variance
            self.counter += 1

            difference = reward - self.reward_mean
            self.reward_mean += difference / self.counter
            new_difference = reward - self.reward_mean
            self.squared_distance_from_mean += difference * new_difference

            if self.counter > 1:
                standard_deviation = (self.squared_distance_from_mean / (self.counter - 1)).sqrt()

            rewards[reward_index] = (reward - self.reward_mean) / standard_deviation

        return rewards

    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        with torch.no_grad():
            rewards = torch.empty(len(self.reward_models), state.shape[0]).to(self.device)
            state = torch.as_tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float).unsqueeze(0)

            for model_index, reward_model in enumerate(self.reward_models):
                if reward_model.ensemble_count > 1:
                    model_state = state.expand(reward_model.ensemble_count, *state.shape[1:])
                    model_actions = actions.expand(reward_model.ensemble_count, *actions.shape[1:])
                    model_rewards = reward_model(
                        model_state,
                        model_actions,
                    ).view(reward_model.ensemble_count, -1, 1)
                    rewards[model_index] = torch.mean(model_rewards, dim=0).squeeze(-1)
                else:
                    rewards[model_index] = reward_model(state, actions).squeeze(1)

            rewards = self.standardize_rewards(rewards.transpose(0, 1))

            # Weight the reward predictions by the inverse of the standard deviation of the models
            if self.inverse_scaling:
                inverse_standard_deviations = 1 / rewards.std(dim=1, keepdim=True)
                weighted_rewards = (inverse_standard_deviations * rewards).sum(dim=1) / inverse_standard_deviations.sum(dim=1)
            else:
                weighted_rewards = torch.mean(rewards, dim=1)

            return weighted_rewards.cpu().numpy()


def main():
    """Run RL agent training."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        default=["evaluative"],
        help="Types of feedback to train the reward model (space-separated list)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=int(1e6),
        help="Number of steps to generate feedback for",
    )
    parser.add_argument(
        "--inverse-scaling",
        action="store_true",
        help="Use inverse scaling for reward weighting",
    )

    args = parser.parse_args()

    FEEDBACK_ID = "_".join([args.algorithm, args.environment, str(args.seed)])
    MODEL_ID = f"{FEEDBACK_ID}_{args.seed}"

    reward_model_paths = []
    for feedback_type in args.feedback_types:
        model_path = os.path.join(
            args.reward_model_folder,
            f"{FEEDBACK_ID}_{feedback_type}_{args.seed}.ckpt",
        )
        reward_model_paths.append(model_path)

    set_random_seed(args.seed)

    run = wandb.init(
        name="ENSEMBLE_RL_" + MODEL_ID,
        project=args.wandb_project_name,
        config={
            **vars(args),
            "feedback_types": args.feedback_types,
            "feedback_count": len(args.feedback_types),
            "total_ensemble_size": len(args.feedback_types),
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # ================ Load correct reward function model =================
    if "ALE/" in args.environment or "procgen" in args.environment:
        architecture_cls = SingleCnnNetwork
    else:
        architecture_cls = SingleNetwork

    # ================ Load correct reward function model ===================
    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join(args.save_folder, f"ENSEMBLE_RL_{MODEL_ID}"),
        tensorboard_log=f"runs/{'ENSEMBLE_RL_'+MODEL_ID}",
        n_timesteps=args.train_steps,
        seed=args.seed,
        log_interval=-1,
        reward_function=CustomReward(
            reward_model_cls=architecture_cls,
            reward_model_paths=reward_model_paths,
            inverse_scaling=args.inverse_scaling,
            device=DEVICE,
        ),
        use_wandb_callback=True,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        # we need to save the loaded hyperparameters
        args.saved_hyperparams = saved_hyperparams
        assert run is not None  # make mypy happy
        run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()

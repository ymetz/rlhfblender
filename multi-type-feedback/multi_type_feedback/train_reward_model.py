"""Module for training a reward model from the generated feedback."""

import math
import os
from typing import Union

import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from multi_type_feedback.datatypes import FeedbackType
from multi_type_feedback.feedback_dataset import FeedbackDataset, LoadFeedbackDataset
from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


def train_reward_model(
    reward_model: LightningModule,
    reward_model_id: str,
    feedback_type: FeedbackType,
    dataset: FeedbackDataset,
    maximum_epochs: int = 100,
    environment: str = "HalfCheetah-v3",
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
    num_ensemble_models: int = 4,
    noise_level: float = 0.0,
    n_feedback: int = -1,
    seed: int = 0,
    wandb_project_name: str = "multi-type-rlhf",
    save_path: str = "reward_models",
):
    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(dataset, lengths=[training_set_size, len(dataset) - training_set_size])

    train_loader = DataLoader(
        train_set,
        batch_size=num_ensemble_models,
        shuffle=True,
        pin_memory=True,
        # num_workers=cpu_count,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=num_ensemble_models,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=reward_model_id,
        config={
            "feedback_type": feedback_type,
            "noise_level": noise_level,
            "seed": seed,
            "environment": environment,
            "n_feedback": n_feedback,
        },
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        accumulate_grad_batches=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            checkpoint_callback,
            *([callback] if callback is not None else []),
        ],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    wandb.finish()

    return reward_model


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument("--feedback-type", type=str, default="evaluative", help="Type of feedback")
    parser.add_argument("--n-ensemble", type=int, default=4, help="Number of ensemble models")
    parser.add_argument("--no-loading-bar", action="store_true", help="Disable loading bar")
    parser.add_argument("--feedback-folder", type=str, default="feedback", help="Folder to load feedback from")
    parser.add_argument("--save-folder", type=str, default="reward_models", help="Save folder for trained reward models")
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    environment = TrainingUtils.setup_environment(args.environment, args.seed, save_reset_wrapper=False)

    feedback_id, model_id = TrainingUtils.get_model_ids(args)

    # Setup reward model
    reward_model = (
        SingleCnnNetwork
        if "procgen" in args.environment or "ALE" in args.environment
        else SingleNetwork
    )(
        input_spaces=(environment.observation_space, environment.action_space),
        hidden_dim=256,
        action_hidden_dim=(16 if "procgen" in args.environment or "ALE" in args.environment else 32),
        layer_num=(3 if "procgen" in args.environment or "ALE" in args.environment else 6),
        cnn_channels=((16, 32, 32) if "procgen" in args.environment or "ALE" in args.environment else None),
        output_dim=1,
        loss_function=(
            calculate_single_reward_loss if args.feedback_type in ["evaluative", "descriptive"] else calculate_pairwise_loss
        ),
        learning_rate=1e-5,
        ensemble_count=args.n_ensemble,
    )

    dataset = LoadFeedbackDataset(
        os.path.join(args.feedback_folder, f"{feedback_id}.pkl"),
        args.feedback_type,
        args.n_feedback,
        noise_level=args.noise_level,
        env=environment if args.feedback_type in "demonstrative" else None,
        env_name=args.environment,
        seed=args.seed,
    )

    train_reward_model(
        reward_model,
        model_id,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        split_ratio=0.85,
        environment=args.environment,
        cpu_count=os.cpu_count() or 8,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=not args.no_loading_bar,
        noise_level=args.noise_level,
        n_feedback=args.n_feedback,
        seed=args.seed,
        wandb_project_name=args.wandb_project_name,
        save_path=args.save_folder,
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Add your project to the path
sys.path.append(os.getcwd())

# Import your project modules
from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils


def update_status(status_file, status, message=""):
    """Update the status file."""
    with open(status_file, "w") as f:
        json.dump({"status": status, "message": message, "timestamp": time.time()}, f)
    print(f"Status updated to {status}: {message}")


def update_result(result_file, **kwargs):
    """Update the result file."""
    # First read existing results
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
    else:
        result = {
            "uncertainty": 0.0,
            "avg_predicted_reward": 0.0,
            "training_loss": 0.0,
            "validation_loss": 0.0,
            "training_complete": False,
            "trained_model_path": "",
            "new_episodes": [],
            "metrics": {},
        }

    # Update with new values
    for key, value in kwargs.items():
        result[key] = value

    # Save updated results
    with open(result_file, "w") as f:
        json.dump(result, f)
    print(f"Updated results: {kwargs}")


def train_single_model(reward_model, dataset, model_id, reward_models_dir, status_file, result_file):
    """
    Train a single reward model and return the path to the saved model.

    Args:
        reward_model: The model to train
        dataset: Training dataset
        model_id: Unique identifier for the model
        reward_models_dir: Directory to save models
        status_file: Status file for updates
        result_file: Result file for updates

    Returns:
        Path to the trained model or None if training failed
    """
    try:
        # Create model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(reward_models_dir),
            filename=model_id,
            monitor="val_loss",
        )

        # Initialize trainer with early stopping
        trainer = Trainer(
            max_epochs=30,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=5,
            gradient_clip_val=None,
            enable_progress_bar=True,
            logger=None,  # Disable wandb for background process
            accumulate_grad_batches=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=1),
                checkpoint_callback,
            ],
        )

        # Create data loaders
        import math

        from torch.utils.data import DataLoader, random_split

        split_ratio = 0.8
        training_set_size = math.floor(split_ratio * len(dataset))
        train_set, val_set = random_split(dataset, lengths=[training_set_size, len(dataset) - training_set_size])

        train_loader = DataLoader(
            train_set,
            batch_size=4,  # num_ensemble_models
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=4,  # num_ensemble_models
            pin_memory=True,
            num_workers=1,
            drop_last=False,
        )

        print(len(val_loader), len(train_loader))

        # Train model
        trainer.fit(reward_model, train_loader, val_loader)

        # Update training results
        train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
        val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))

        update_result(
            result_file,
            training_loss=train_loss,
            validation_loss=val_loss,
        )

        return checkpoint_callback.best_model_path

    except Exception as e:
        error_msg = f"Error during training: {e!s}\n{traceback.format_exc()}"
        print(error_msg)
        update_status(status_file, "failed", error_msg)
        return None


def main():
    parser = argparse.ArgumentParser(description="Train reward model from feedback")
    parser.add_argument("--feedback-file", type=str, required=True, help="Path to feedback file")
    parser.add_argument("--session-dir", type=str, required=True, help="Path to session directory")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    status_file = session_dir / "status.json"
    result_file = session_dir / "result.json"

    try:

        # Set up environment
        update_status(status_file, "initializing", "Setting up environment")
        print(args.seed)
        TrainingUtils.set_seeds(args.seed)
        environment = TrainingUtils.setup_environment(args.env, args.seed, save_reset_wrapper=False)

        print(f"Environment {args.env} initialized with seed {args.seed}")

        # Load preprocessed datasets by type
        update_status(status_file, "processing_feedback", "Loading preprocessed datasets")

        # Get feedback types from dataset files
        feedback_types = set()
        for file_path in session_dir.glob("*_dataset.pkl"):
            fb_type = file_path.stem.replace("_dataset", "")
            feedback_types.add(fb_type)

        if not feedback_types:
            raise FileNotFoundError(
                f"No preprocessed datasets found in {session_dir}. RewardModelHandler should have created these."
            )

        print(f"Found feedback types: {feedback_types}")
        # Train individual reward models for each feedback type
        reward_models = []
        all_model_paths = []
        reward_models_dir = session_dir / "reward_models"
        reward_models_dir.mkdir(exist_ok=True)

        for feedback_type in feedback_types:
            update_status(status_file, "training", f"Training reward model for {feedback_type} feedback")

            # Load dataset for this feedback type
            dataset_file = session_dir / f"{feedback_type}_dataset.pkl"
            with open(dataset_file, "rb") as f:
                dataset = pickle.load(f)
            print(f"Loaded {feedback_type} dataset from {dataset_file}")

            # Set up reward model for this feedback type
            is_cnn_env = "procgen" in args.env or "ALE" in args.env
            model_kwargs = {
                "input_spaces": (environment.observation_space, environment.action_space),
                "hidden_dim": 256,
                "action_hidden_dim": (16 if is_cnn_env else 32),
                "layer_num": (3 if is_cnn_env else 6),
                "output_dim": 1,
                "loss_function": (
                    calculate_single_reward_loss if feedback_type in ["rating", "descriptive"] else calculate_pairwise_loss
                ),
                "learning_rate": 1e-5,
                "ensemble_count": 4,  # Use ensemble of 4 models
            }

            if is_cnn_env:
                model_kwargs["cnn_channels"] = [16, 32, 32]
                reward_model = SingleCnnNetwork(**model_kwargs)
            else:
                reward_model = SingleNetwork(**model_kwargs)

            # Train the model for this feedback type
            model_id = f"{args.algorithm}_{args.env}_{args.seed}_{feedback_type}_{args.seed}"

            # Train and save the model
            trained_model_path = train_single_model(
                reward_model, dataset, model_id, reward_models_dir, status_file, result_file
            )

            if trained_model_path:
                all_model_paths.append(trained_model_path)
                reward_models.append(reward_model)

        # Update with all model paths
        update_result(
            result_file,
            trained_model_path=all_model_paths[0] if all_model_paths else "",
            metrics={"model_paths": all_model_paths},
        )

        # Calculate basic metrics from training
        update_status(status_file, "evaluating", "Finalizing training results")

        # Basic metrics from the training process
        uncertainty = 0.0
        avg_reward = 0.0

        # Calculate total samples across all datasets
        total_samples = 0
        for feedback_type in feedback_types:
            dataset_file = session_dir / f"{feedback_type}_dataset.pkl"
            if dataset_file.exists():
                with open(dataset_file, "rb") as f:
                    dataset = pickle.load(f)
                    if hasattr(dataset, "__len__"):
                        total_samples += len(dataset)

        # Read existing results to preserve training metrics
        existing_result = {}
        if os.path.exists(result_file):
            with open(result_file) as f:
                existing_result = json.load(f)

        # Merge existing metrics with new ones
        existing_metrics = existing_result.get("metrics", {})
        new_metrics = {
            "total_dataset_samples": total_samples,
            "feedback_types": list(feedback_types),
            "num_models_trained": len(all_model_paths),
            "model_paths": all_model_paths,
            "individual_training": True,
        }
        merged_metrics = {**existing_metrics, **new_metrics}

        # Update final results
        update_result(
            result_file,
            uncertainty=uncertainty,
            avg_predicted_reward=avg_reward,
            new_episodes=[],  # No new episodes generated, using preprocessed data
            training_complete=True,
            metrics=merged_metrics,
        )

        update_status(status_file, "completed", "Training completed successfully")

    except Exception as e:
        import traceback

        error_msg = f"Error during training: {e!s}\\n{traceback.format_exc()}"
        print(error_msg)
        update_status(status_file, "failed", error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

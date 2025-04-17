import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Add your project to the path
sys.path.append(os.getcwd())

# Import your project modules
from multi_type_feedback.datatypes import FeedbackType
from multi_type_feedback.feedback_dataset import FeedbackDataset, LoadFeedbackDataset
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
        with open(result_file, "r") as f:
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
        TrainingUtils.set_seeds(args.seed)
        environment = TrainingUtils.setup_environment(args.env, save_reset_wrapper=False)

        # Load feedback
        update_status(status_file, "processing_feedback", "Loading feedback data")
        with open(args.feedback_file, "rb") as f:
            feedback_data = pickle.load(f)

        # Identify feedback types in the data
        feedback_types = set()
        for item in feedback_data:
            if "feedback_type" in item:
                feedback_types.add(item["feedback_type"])
            else:
                # Default to evaluative if not specified
                feedback_types.add("evaluative")

        # Create feedback datasets for each type
        reward_models = []
        all_model_paths = []

        for feedback_type in feedback_types:
            update_status(status_file, "training", f"Training reward model for {feedback_type} feedback")

            # Filter feedback by type
            type_feedback = [item for item in feedback_data if item.get("feedback_type", "evaluative") == feedback_type]

            # Save type-specific feedback
            type_feedback_file = session_dir / f"{feedback_type}_feedback.pkl"
            with open(type_feedback_file, "wb") as f:
                pickle.dump(type_feedback, f)

            # Create dataset
            dataset = LoadFeedbackDataset(
                str(type_feedback_file),
                feedback_type,
                -1,  # Use all feedback
                noise_level=0.0,
                env=environment if feedback_type == "demonstrative" else None,
                env_name=args.env,
                seed=args.seed,
            )

            # Set up reward model
            reward_model = (SingleCnnNetwork if "procgen" in args.env or "ALE" in args.env else SingleNetwork)(
                input_spaces=(environment.observation_space, environment.action_space),
                hidden_dim=256,
                action_hidden_dim=(16 if "procgen" in args.env or "ALE" in args.env else 32),
                layer_num=(3 if "procgen" in args.env or "ALE" in args.env else 6),
                cnn_channels=((16, 32, 32) if "procgen" in args.env or "ALE" in args.env else None),
                output_dim=1,
                loss_function=(
                    calculate_single_reward_loss if feedback_type in ["evaluative", "descriptive"] else calculate_pairwise_loss
                ),
                learning_rate=1e-5,
                ensemble_count=4,  # Use ensemble of 4 models
            )

            # Create model ID
            model_id = f"{args.algorithm}_{args.env}_{args.seed}_{feedback_type}_{args.seed}"

            # Create reward models directory if it doesn't exist
            reward_models_dir = session_dir / "reward_models"
            reward_models_dir.mkdir(exist_ok=True)

            # Train reward model
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(reward_models_dir),
                filename=model_id,
                monitor="val_loss",
            )

            # Initialize trainer with early stopping
            trainer = Trainer(
                max_epochs=100,
                accelerator="auto",
                devices="auto",
                log_every_n_steps=5,
                gradient_clip_val=None,
                enable_progress_bar=True,
                logger=None,  # Disable wandb for background process
                accumulate_grad_batches=32,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=5),
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
                drop_last=True,
            )

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

            # Save best model path
            best_model_path = checkpoint_callback.best_model_path
            all_model_paths.append(best_model_path)

            # Collect model for ensemble
            reward_models.append(reward_model)

        # Update with all model paths
        update_result(
            result_file,
            trained_model_path=all_model_paths[0] if all_model_paths else "",
            metrics={"model_paths": all_model_paths},
        )

        # Collect episodes with the trained reward model
        update_status(status_file, "collecting_episodes", "Collecting new episodes with trained model")

        model_path = all_model_paths[0] if all_model_paths else ""
        episode_cmd = [
            "python",
            os.path.join(os.path.dirname(__file__), "collect_episodes.py"),
            "--session-dir",
            str(session_dir),
            "--env",
            args.env,
            "--num-episodes",
            "10",
            "--random-policy",
            "false" if model_path else "true",
            "--reward-model-path",
            model_path if model_path else "",
        ]

        # Run process and wait for completion
        episode_log = open(session_dir / "episode_collection.log", "w")
        episode_process = subprocess.Popen(episode_cmd, stdout=episode_log, stderr=subprocess.STDOUT)

        episode_process.wait()

        # Load episodes
        episodes_file = session_dir / "episodes.json"
        episodes = []
        if os.path.exists(episodes_file):
            with open(episodes_file, "r") as f:
                episodes = json.load(f)

        # Calculate uncertainty and average predicted reward
        update_status(status_file, "evaluating", "Evaluating trained reward model")

        # Calculate uncertainty from ensemble predictions if available
        uncertainty = 0.0
        avg_reward = 0.0

        if reward_models and len(reward_models) > 0 and episodes:
            # Get predictions from all models for a sample of states/actions
            predictions = []

            # Use a subset of episode data for evaluation
            eval_observations = []
            eval_actions = []
            for ep in episodes[: min(len(episodes), 5)]:  # Use up to 5 episodes
                # Sample up to 20 states per episode
                for i in range(0, min(len(ep["observations"]), 100), 5):
                    eval_observations.append(ep["observations"][i])
                    eval_actions.append(ep["actions"][i])

            # Convert to tensors
            if eval_observations and eval_actions:
                with torch.no_grad():
                    # Convert to tensors
                    obs = torch.tensor(eval_observations, dtype=torch.float32)
                    actions = torch.tensor(eval_actions, dtype=torch.float32)

                    # Get predictions from each model
                    model_predictions = []
                    for model in reward_models:
                        # For ensemble models, we get multiple predictions per input
                        if model.ensemble_count > 1:
                            # Expand inputs for ensemble
                            ensemble_obs = obs.unsqueeze(0).expand(model.ensemble_count, *obs.shape)
                            ensemble_actions = actions.unsqueeze(0).expand(model.ensemble_count, *actions.shape)

                            # Get predictions from ensemble
                            preds = model(ensemble_obs, ensemble_actions).view(model.ensemble_count, -1)

                            # Store all ensemble predictions separately
                            for i in range(model.ensemble_count):
                                model_predictions.append(preds[i].cpu().numpy())
                        else:
                            preds = model(obs, actions).squeeze(-1)
                            model_predictions.append(preds.cpu().numpy())

                    # Calculate uncertainty (standard deviation across models)
                    model_predictions = np.array(model_predictions)
                    uncertainty = float(np.mean(np.std(model_predictions, axis=0)))
                    avg_reward = float(np.mean(model_predictions))

        # Update final results
        update_result(
            result_file,
            uncertainty=uncertainty,
            avg_predicted_reward=avg_reward,
            new_episodes=episodes,
            training_complete=True,
            metrics={
                "num_feedback": len(feedback_data),
                "feedback_types": list(feedback_types),
                "num_episodes": len(episodes),
                "model_paths": all_model_paths,
            },
        )

        update_status(status_file, "completed", "Training completed successfully")

    except Exception as e:
        import traceback

        error_msg = f"Error during training: {str(e)}\\n{traceback.format_exc()}"
        print(error_msg)
        update_status(status_file, "failed", error_msg)
        sys.exit(1)

import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from databases import Database
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from multi_type_feedback.dynamic_rlhf_human import DynamicRLHF
from pydantic import BaseModel
from train_baselines.exp_manager import ExperimentManager

from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.global_models import Experiment
from rlhfblender.utils import process_env_name
from rlhfblender.utils.data_generation import _process_benchmark_data

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

router = APIRouter(prefix="/dynamic_rlhf")

# Import shared store for active human-in-the-loop training sessions
from rlhfblender.routes.data import active_rlhf_sessions

# Simulation store for web demo
simulation_store = {}


def save_trajectories_to_data_dir(
    trajectories: list[list[tuple[np.ndarray, np.ndarray, float, bool, float]]],
    initial_states: list[Any],
    env_name: str,
    exp_id: str,
    checkpoint_step: int,
) -> str:
    """
    Save trajectories to the data directory in the format compatible with rlhfblender.
    Creates a consolidated file similar to episode_recorder.py for use with _process_benchmark_data.

    Args:
        trajectories: List of trajectories, each containing (obs, action, reward, done, uncertainty) tuples
        initial_states: List of initial states for each trajectory
        env_name: Environment name
        exp_id: Experiment ID
        checkpoint_step: Current checkpoint step (0 for initial collection)

    Returns:
        Path where the consolidated trajectory file was saved
    """
    # Create buffers similar to episode_recorder
    buffers = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
        "probs": [],
        "renders": [],
        "uncertainty": [],  # Add uncertainty buffer
    }
    episode_rewards = []
    episode_lengths = []

    # Process each trajectory
    for episode_idx, (trajectory, initial_state) in enumerate(zip(trajectories, initial_states)):
        if not trajectory:
            continue

        # Extract components from trajectory
        for step_idx, step_data in enumerate(trajectory):
            if len(step_data) == 6:
                obs, action, reward, done, uncertainty, render = step_data
            else:
                # Fallback for trajectory without render data
                obs, action, reward, done, uncertainty = step_data
                render = np.zeros((128, 128, 3), dtype=np.uint8)
            is_last_step = step_idx == len(trajectory) - 1
            buffers["obs"].append(np.squeeze(obs))
            buffers["actions"].append(np.squeeze(action))
            buffers["rewards"].append(reward)
            buffers["dones"].append(is_last_step or done)  # Use is_last_step for last step
            buffers["uncertainty"].append(uncertainty)  # Add uncertainty data

            # Create info with basic information including uncertainty
            info = {"timestep": step_idx, "episode_id": episode_idx}
            buffers["infos"].append(info)

            # Create placeholder arrays for compatibility
            buffers["probs"].append(0.0)  # Placeholder for action probabilities
            buffers["renders"].append(render)

        # Calculate episode metrics
        episode_reward = sum(step[2] for step in trajectory)
        episode_length = len(trajectory)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    for key in buffers:
        if key == "renders":
            buffers[key] = np.array(buffers[key])
        else:
            buffers[key] = np.array(buffers[key])

    env_name_processed = process_env_name(env_name)
    save_file_name = os.path.join(env_name_processed, f"{env_name_processed}_{exp_id}_{checkpoint_step}")
    save_path = os.path.join("data", "saved_benchmarks", save_file_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path + ".npz", "wb") as f:
        np.savez(
            f,
            obs=buffers["obs"],
            actions=buffers["actions"],
            rewards=buffers["rewards"],
            dones=buffers["dones"],
            infos=buffers["infos"],
            probs=buffers["probs"],
            renders=buffers["renders"],
            uncertainty=buffers["uncertainty"],
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            additional_metrics={},
        )

    print(f"Saved {len(trajectories)} trajectories to consolidated file: {save_path}.npz")
    return save_path + ".npz"


def save_feedback_status(
    trajectories: list[list[tuple[np.ndarray, np.ndarray, float, bool, float]]],
    session_id: str,
    phase: int,
    status: str = "candidates_ready",
) -> None:
    """
    Save feedback status with predicted rewards and uncertainties from trajectories.

    Args:
        trajectories: List of trajectories containing (obs, action, reward, done, uncertainty) tuples
        session_id: Session ID for the feedback status file
        phase: Current training phase
        status: Status to save (default: "candidates_ready")
    """
    try:
        # Extract rewards and uncertainties from trajectories
        all_rewards = []
        all_uncertainties = []

        for trajectory in trajectories:
            if not trajectory:
                continue

            trajectory_rewards = []
            trajectory_uncertainties = []

            for step_data in trajectory:
                if len(step_data) >= 5:
                    # (obs, action, reward, done, uncertainty, ...)
                    reward = step_data[2]
                    uncertainty = step_data[4]
                    trajectory_rewards.append(reward)
                    trajectory_uncertainties.append(uncertainty)

            if trajectory_rewards:
                # Use sum of rewards for trajectory total reward
                trajectory_total_reward = sum(trajectory_rewards)
                # Use mean uncertainty for trajectory uncertainty
                trajectory_mean_uncertainty = np.mean(trajectory_uncertainties)

                all_rewards.append(trajectory_total_reward)
                all_uncertainties.append(trajectory_mean_uncertainty)

        # Compute averages across all trajectories
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        avg_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0

        # Create feedback status data
        feedback_status = {
            "status": status,
            "phase": phase,
            "avg_predicted_reward": float(avg_reward),
            "avg_uncertainty": float(avg_uncertainty),
            "num_trajectories": len(trajectories),
            "timestamp": time.time(),
        }

        # Ensure sessions directory exists
        sessions_dir = Path(f"sessions/{session_id}")
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Save feedback status
        feedback_status_file = sessions_dir / "feedback_status.json"
        with open(feedback_status_file, "w") as f:
            json.dump(feedback_status, f, indent=2)

        print(f"Saved feedback status: avg_reward={avg_reward:.3f}, avg_uncertainty={avg_uncertainty:.3f}")

    except Exception as e:
        print(f"Error saving feedback status: {e}")


async def process_dynamic_rlhf_trajectories(env_name: str, exp_id: str, checkpoint_step: int) -> bool:
    """
    Process DynamicRLHF trajectories to generate videos, rewards, uncertainty, and thumbnails.

    Args:
        env_name: Environment name
        exp_id: Experiment ID
        checkpoint_step: Checkpoint step number

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Create a benchmark request dict similar to the format used in data_generation
        benchmark_request = {
            "env": env_name,
            "checkpoint_step": checkpoint_step,
        }

        # Create the expected path structure for the benchmark data
        env_name_processed = process_env_name(env_name)
        benchmark_file = os.path.join(
            "data", "saved_benchmarks", env_name_processed, f"{env_name_processed}_{exp_id}_{checkpoint_step}.npz"
        )

        # Check if benchmark file exists
        if not os.path.exists(benchmark_file):
            print(f"Benchmark file not found: {benchmark_file}")
            return False

        print(f"Processing benchmark file: {benchmark_file}")

        # Create the request and call the processing function
        requests = [benchmark_request]
        benchmarked_experiments = [exp_id]

        # Call the existing processing function
        await _process_benchmark_data(requests, benchmarked_experiments)

        print(f"Successfully processed DynamicRLHF trajectories for {env_name}_{exp_id}_{checkpoint_step}")
        return True

    except Exception as e:
        print(f"Error processing DynamicRLHF trajectories: {e}")
        import traceback

        traceback.print_exc()
        return False


async def generate_dynamic_rlhf_projections(env_name: str, exp_id: str, checkpoint_step: int) -> bool:
    """
    Generate projections and uncertainty maps for DynamicRLHF visualization.

    Args:
        env_name: Environment name
        exp_id: Experiment ID
        checkpoint_step: Checkpoint step number

    Returns:
        True if generation was successful, False otherwise
    """
    import subprocess
    import sys

    try:
        # Get experiment from database to find the correct experiment name
        exp: Experiment = await db_handler.get_single_entry(database, Experiment, key=exp_id)
        if not exp:
            print(f"Could not find experiment with ID: {exp_id}")
            return False

        experiment_name = exp.exp_name
        env_name_processed = process_env_name(env_name)

        # Path to episode data
        episode_path = os.path.join("data", "episodes", env_name_processed, f"{env_name_processed}_{exp_id}_{checkpoint_step}")

        # Check if episode data exists
        if not os.path.exists(episode_path):
            print(f"Episode data not found: {episode_path}")
            return False

        print(f"Generating projections for experiment {experiment_name}, checkpoint {checkpoint_step}")

        # 1. Generate projections (using PCA as default for consistency)
        projection_cmd = [
            sys.executable,
            "rlhfblender/projections/generate_projections.py",
            "--experiment-name",
            experiment_name,
            "--checkpoint",
            str(checkpoint_step),
            "--projection-method",
            "PCA",
            "--compute-inverse",
            "--auto-grid-range",
            "--grid-resolution",
            "20",
        ]

        print(f"Running projection generation: {' '.join(projection_cmd)}")
        projection_result = subprocess.run(projection_cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout

        if projection_result.returncode != 0:
            print(f"Projection generation failed: {projection_result.stderr}")
            return False

        print("Projection generation completed successfully")

        # 2. Generate reward and uncertainty predictions
        # We need to find the DynamicRLHF session that contains this experiment
        session = None
        session_id_found = None
        for sid, sess_data in active_rlhf_sessions.items():
            if str(sess_data.get("experiment_id")) == str(exp_id):
                session = sess_data
                break

        if session and "drlhf" in session:
            # Get saved model paths from the session
            saved_models = session.get("last_saved_models", {})
            saved_agent = session.get("last_saved_agent", "")

            print("SAVED MODELS:", saved_models, saved_agent)

            if saved_models and saved_agent:
                # Find the projection file that was just created
                # The projection script uses the format: {env_name_processed}_{db_experiment.id}_{checkpoint_step}_{projection_method}
                projection_file = os.path.join(
                    "data", "saved_projections", f"{env_name_processed}_{exp_id}_{checkpoint_step}_PCA.json"
                )

                # Use the first available reward model (for separate models) or the unified model
                reward_model_path = None
                if "unified" in saved_models:
                    reward_model_path = saved_models["unified"]
                elif saved_models:
                    # Take the first available model for separate models
                    reward_model_path = list(saved_models.values())[0]

                print(f"Checking required files for prediction:")
                print(
                    f"  Reward model: {reward_model_path} (exists: {os.path.exists(reward_model_path) if reward_model_path else False})"
                )
                print(f"  Policy model: {saved_agent} (exists: {os.path.exists(saved_agent)})")
                print(f"  Projection file: {projection_file} (exists: {os.path.exists(projection_file)})")
                print(f"  Episode path: {episode_path} (exists: {os.path.exists(episode_path)})")

                if (
                    reward_model_path
                    and os.path.exists(reward_model_path)
                    and os.path.exists(saved_agent)
                    and os.path.exists(projection_file)
                    and os.path.exists(episode_path)
                ):
                    # Generate reward and uncertainty predictions
                    prediction_cmd = [
                        sys.executable,
                        "rlhfblender/projections/predict_reward_and_uncertainty.py",
                        "--experiment-name",
                        experiment_name,
                        "--reward-model",
                        reward_model_path,
                        "--output-dir",
                        "data/saved_projections",
                        "--policy-model",
                        saved_agent,
                        "--projection-data",
                        projection_file,
                        "--episode-path",
                        episode_path,
                    ]

                    print(f"Running reward/uncertainty prediction: {' '.join(prediction_cmd)}")
                    prediction_result = subprocess.run(
                        prediction_cmd, capture_output=True, text=True, timeout=180  # 3 minute timeout
                    )

                    if prediction_result.returncode != 0:
                        print(f"Reward/uncertainty prediction failed: {prediction_result.stderr}")
                        return True  # Still consider successful since projections were generated

                    print("Reward/uncertainty prediction completed successfully")
                    return True
                else:
                    print(f"Some required files not found - skipping reward/uncertainty prediction")
                    return True
            else:
                print("No saved models found for reward/uncertainty prediction")
                return True  # Still consider successful since projections were generated
        else:
            print("DynamicRLHF session not found - skipping reward/uncertainty prediction")
            return True  # Still consider successful since projections were generated

    except subprocess.TimeoutExpired:
        print("Projection generation timed out")
        return False
    except Exception as e:
        print(f"Error generating projections: {e}")
        import traceback

        traceback.print_exc()
        return False


async def initialize_dynamic_rlhf_session(
    session_id: str, experiment_id: str, num_iterations: int = 5, resume_from_checkpoint: int = None
) -> dict[str, Any]:
    """
    Initialize a DynamicRLHF session for the given experiment.
    Returns the session data or raises an exception if initialization fails.
    """
    # Get experiment from database
    exp: Experiment = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        raise ValueError("No experiment found")

    # Create ExperimentManager for proper hyperparameter loading
    exp_manager = ExperimentManager(
        args=SimpleNamespace(), algo=exp.algorithm.lower(), env_id=exp.env_id, log_folder=f"dynamic_rlhf_models/{session_id}"
    )

    # Get hyperparameters and total timesteps from ExperimentManager
    hyperparams = exp_manager.get_hyperparam_config_for_algo()
    total_timesteps = exp_manager.n_timesteps if exp_manager.n_timesteps > 0 else hyperparams.get("n_timesteps", 1000000)

    # Calculate RL steps per iteration based on total timesteps and number of iterations
    rl_steps_per_iteration = total_timesteps // num_iterations

    # Get algorithm and environment configuration from experiment
    env_kwargs = exp.environment_config.get("env_kwargs", {})

    # Initialize DynamicRLHF with ExperimentManager and calculated parameters
    drlhf = DynamicRLHF(
        oracle=None,  # No oracle for human feedback
        env_name=exp.env_id,
        env_kwargs=env_kwargs,
        algorithm=exp.algorithm,
        feedback_types=["evaluative", "comparative", "demonstrative", "corrective"],
        n_training_iterations=num_iterations,
        n_feedback_per_iteration=10,
        feedback_buffer_size=1000,
        rl_steps_per_iteration=rl_steps_per_iteration,
        reward_training_epochs=10,
        device="cpu",
        num_ensemble_models=2,
        initial_feedback_count=20,
        seed=42,
        exp_manager=exp_manager,
    )

    # Load existing models if resuming from a checkpoint
    has_initial_data = False
    if resume_from_checkpoint is not None and resume_from_checkpoint > 0:
        print(f"Attempting to resume from checkpoint {resume_from_checkpoint}")
        loaded_drlhf = await load_full_checkpoint(session_id, exp, resume_from_checkpoint)
        if loaded_drlhf is not None:
            drlhf = loaded_drlhf  # Replace the fresh instance with the loaded one
            has_initial_data = True
            print(f"Successfully loaded complete DynamicRLHF state from checkpoint {resume_from_checkpoint}")
        else:
            print(f"Warning: Could not load checkpoint {resume_from_checkpoint}, starting fresh")
    else:
        print("Starting with fresh untrained models")

    # Create session data
    session_data = {
        "drlhf": drlhf,
        "experiment_id": exp.id,
        "env_name": exp.env_id,
        "algorithm": exp.algorithm,
        "status": "initialized",
        "created_at": time.time(),
        "phase": resume_from_checkpoint if resume_from_checkpoint is not None else 0,
        "training_step": resume_from_checkpoint if resume_from_checkpoint is not None else 0,
        "save_path": f"dynamic_rlhf_models/{session_id}",
        "session_id": session_id,
        "has_initial_data": has_initial_data,
        "resume_from_checkpoint": resume_from_checkpoint,
    }

    return session_data


async def load_full_checkpoint(session_id: str, experiment: Experiment, checkpoint: int):
    """
    Load a complete DynamicRLHF checkpoint using the existing save/load functionality.

    Args:
        session_id: Session ID for path construction
        experiment: Experiment object
        checkpoint: Checkpoint number to load from

    Returns:
        Loaded DynamicRLHF instance or None if loading failed
    """
    try:
        from types import SimpleNamespace

        from multi_type_feedback.dynamic_rlhf_human import DynamicRLHF
        from train_baselines.exp_manager import ExperimentManager

        # Construct the checkpoint path - this should match where save() puts the files
        checkpoint_path = f"dynamic_rlhf_models/{session_id}_checkpoint_{checkpoint}"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint directory not found: {checkpoint_path}")
            return None

        print(f"Loading DynamicRLHF checkpoint from: {checkpoint_path}")

        # Create ExperimentManager for the loaded instance (same as in initialization)
        exp_manager = ExperimentManager(
            args=SimpleNamespace(),
            algo=experiment.algorithm.lower(),
            env_id=experiment.env_id,
            log_folder=f"dynamic_rlhf_models/{session_id}",
        )

        # Use the existing load method
        drlhf = DynamicRLHF.load(
            load_path=checkpoint_path, oracle=None, exp_manager=exp_manager  # No oracle for human feedback
        )

        print(f"Successfully loaded DynamicRLHF checkpoint {checkpoint}")
        return drlhf

    except Exception as e:
        print(f"Error loading DynamicRLHF checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


@router.post("/start_dynamic_rlhf_training", response_model=dict[str, Any])
async def start_dynamic_rlhf_training(request: Request):
    """
    Start a new DynamicRLHF training session with experiment creation.
    Only available when SIMULATE_TRAINING=false for actual training.
    """
    # Check if we're in actual training mode (not simulation)
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"
    if simulate_training:
        return JSONResponse(
            status_code=403, content={"status": "error", "message": "DynamicRLHF training not available in simulation mode"}
        )

    try:
        request_data = await request.json()
        num_iterations = request_data.get("num_iterations", 5)
        experiment_name = request_data.get("experiment_name")
        environment_id = request_data.get("environment_id")
        clone_from_experiment = request_data.get("clone_from_experiment")
        resume_from_checkpoint = request_data.get("resume_from_checkpoint", None)

        if not experiment_name:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing experiment_name"})

        if not clone_from_experiment and not environment_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Either clone_from_experiment or environment_id must be provided"},
            )

        # Import necessary modules
        import uuid

        from rlhfblender.data_models.global_models import Experiment
        from rlhfblender.utils.data_generation import register_experiment

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Clone experiment if requested
        if clone_from_experiment:
            # Get the experiment to clone
            clone_exp = await db_handler.get_single_entry(
                database, Experiment, key=clone_from_experiment, key_column="exp_name"
            )
            if clone_exp:
                # Use environment from cloned experiment
                actual_env_id = clone_exp.env_id
                actual_env_kwargs = clone_exp.environment_config.get("env_kwargs", {})

                # Register new experiment with cloned settings
                await register_experiment(
                    exp_name=experiment_name,
                    env_id=actual_env_id,
                    env_kwargs=actual_env_kwargs,
                    path=f"dynamic_rlhf_models/{session_id}",  # Use session-specific path
                    algorithm=clone_exp.algorithm,
                    framework=clone_exp.framework,
                    project="RLHF-Blender",
                )
            else:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": f"Experiment to clone '{clone_from_experiment}' not found"},
                )
        else:
            # Use provided environment_id for new experiment
            actual_env_id = environment_id
            actual_env_kwargs = {}

            # Register new experiment with default settings
            await register_experiment(
                exp_name=experiment_name,
                env_id=actual_env_id,
                env_kwargs=actual_env_kwargs,
                path=f"dynamic_rlhf_models/{session_id}",
                algorithm="ppo",  # Default algorithm
                framework="StableBaselines3",
                project="RLHF-Blender",
            )

        # Get the created experiment
        exp = await db_handler.get_single_entry(database, Experiment, key=experiment_name, key_column="exp_name")

        if not exp:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to create experiment"})

        # Add checkpoint 0 to the experiment's checkpoint list if not already present
        existing_checkpoints = exp.checkpoint_list if exp.checkpoint_list else []
        if 0 not in existing_checkpoints:
            existing_checkpoints.append(0)
            existing_checkpoints.sort()
            await db_handler.update_entry(
                database,
                Experiment,
                key=exp.id,
                data={"checkpoint_list": existing_checkpoints},
            )

        # Initialize DynamicRLHF session using the helper function
        session_data = await initialize_dynamic_rlhf_session(session_id, str(exp.id), num_iterations, resume_from_checkpoint)
        session_data["experiment_name"] = experiment_name  # Add experiment name for tracking

        # Store the active session
        active_rlhf_sessions[session_id] = session_data

        print(f"DynamicRLHF training session started: {session_id} for experiment '{experiment_name}'")

        return JSONResponse(
            content={
                "status": "success",
                "message": f"DynamicRLHF training started for experiment '{experiment_name}'",
                "session_id": session_id,
                "experiment_id": exp.id,
                "experiment_name": experiment_name,
                "env_name": exp.env_id,
                "algorithm": session_data["algorithm"],
                "model_path": exp.path,
            }
        )

    except Exception as e:
        import traceback

        error_msg = f"Error starting DynamicRLHF training: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        return JSONResponse(
            status_code=500, content={"status": "error", "message": f"Failed to start DynamicRLHF training: {str(e)}"}
        )


class FeedbackItem(BaseModel):
    """Model for feedback items."""

    feedback_type: str
    trajectory_id: int
    value: float
    metadata: dict[str, Any] = {}


async def run_training_iteration_background(
    session_id: str, experiment_id: str, phase: int, active_rlhf_sessions: dict
) -> None:
    """
    Background task that performs the actual training iteration work.
    Updates session status as it progresses.
    """
    try:
        session = active_rlhf_sessions[session_id]
        session["status"] = "training"

        # Get the DynamicRLHF instance and experiment info
        drlhf = session["drlhf"]
        exp_id = session["experiment_id"]
        env_name = session["env_name"]

        # Calculate checkpoint step based on current phase
        checkpoint_step = phase

        # Check session state
        has_initial_data = session.get("has_initial_data", False)
        has_feedback = any(len(drlhf.feedback_buffers[ft]) > 0 for ft in drlhf.feedback_types)

        print(f"Background training: phase={phase}, has_initial_data={has_initial_data}, has_feedback={has_feedback}")

        if phase == 0:
            # Initial data collection with untrained models
            session["status"] = "collecting_initial_data"
            print(f"Initial data collection: Collecting trajectories with untrained models...")
            trajectories, initial_states = drlhf.collect_trajectories(
                n_trajectories=drlhf.initial_feedback_count,
                render=True,
            )

            # Save trajectories to data directory
            save_path = save_trajectories_to_data_dir(
                trajectories=trajectories,
                initial_states=initial_states,
                env_name=env_name,
                exp_id=str(exp_id),
                checkpoint_step=checkpoint_step,
            )

            print(f"Saved initial trajectories to: {save_path}")

            # Save feedback status with predicted rewards and uncertainties for initial data
            save_feedback_status(trajectories=trajectories, session_id=session_id, phase=phase, status="candidates_ready")

            # Save initial (untrained) models using projection-compatible paths
            session["status"] = "saving_models"
            print("Saving initial (untrained) models for checkpoint 0")
            checkpoint_save_path = f"dynamic_rlhf_models/{session_id}_checkpoint_{checkpoint_step}"
            drlhf.save(checkpoint_save_path, checkpoint_step, str(exp_id))
            print(f"Saved initial models to projection-compatible paths for checkpoint {checkpoint_step}")

            # Mark that initial data has been collected
            session["has_initial_data"] = True

        else:
            # Training iteration
            print(f"Phase {phase}: Starting training iteration...")

            # Step 1: Load and integrate human feedback from UI
            session["status"] = "loading_feedback"
            print("Loading processed feedback using DynamicRLHF infrastructure...")

            dynamic_rlhf_dir = f"sessions/{session_id}"
            dynamic_rlhf_file_prefix = os.path.join(dynamic_rlhf_dir, "dynamic_rlhf_feedback_")
            dynamic_rlhf_files = [f for f in os.listdir(dynamic_rlhf_dir) if f.startswith("dynamic_rlhf_feedback_")]
            if not dynamic_rlhf_files:
                print(f"No DynamicRLHF format feedback files found.")
                session["status"] = "error"
                session["error"] = "No feedback files found"
                return

            for dynamic_rlhf_file in dynamic_rlhf_files:
                print(f"Found DynamicRLHF format feedback file: {dynamic_rlhf_file}")

                try:
                    stats = drlhf.load_feedback_dataset(dynamic_rlhf_file_prefix)
                    print(f"Feedback integration stats: {stats}")
                except Exception as e:
                    print(f"Error loading DynamicRLHF feedback: {e}")

            # Step 2: Train reward models with collected feedback
            session["status"] = "training_reward_models"
            print("Training reward models with collected feedback...")
            try:
                reward_metrics = drlhf.train_reward_models()
                print(f"Reward model training completed. Metrics: {reward_metrics}")
            except Exception as e:
                import traceback

                print(f"Warning: Reward model training failed: {e}")
                traceback.print_exc()
                reward_metrics = {}

            # Step 3: Train RL agent with updated reward models
            session["status"] = "training_rl_agent"
            print("Training RL agent with updated reward models...")
            try:
                # Calculate RL training steps for this iteration
                rl_steps = drlhf.rl_steps_per_iteration

                # Train the RL agent
                if drlhf.exp_manager:
                    drlhf.exp_manager.learn(drlhf.rl_agent)
                else:
                    drlhf.rl_agent.learn(
                        total_timesteps=rl_steps,
                        reset_num_timesteps=False,
                    )
                print("RL agent training completed")
            except Exception as e:
                print(f"Warning: RL agent training failed: {e}")

            # Step 4: Collect new trajectories with trained agent
            session["status"] = "collecting_trajectories"
            print("Collecting new trajectories with trained agent...")
            trajectories, initial_states = drlhf.collect_trajectories(
                n_trajectories=drlhf.n_feedback_per_iteration,
                render=True,
            )

            # Save new trajectories to data directory
            save_path = save_trajectories_to_data_dir(
                trajectories=trajectories,
                initial_states=initial_states,
                env_name=env_name,
                exp_id=str(exp_id),
                checkpoint_step=checkpoint_step,
            )

        print(f"Saved trajectories to: {save_path}")

        # Save feedback status with predicted rewards and uncertainties
        save_feedback_status(trajectories=trajectories, session_id=session_id, phase=phase, status="candidates_ready")

        # Store common session info
        session["last_trajectories_path"] = save_path
        session["last_checkpoint_step"] = checkpoint_step

        # Save full DynamicRLHF checkpoint for resumption capability
        session["status"] = "saving_checkpoint"
        checkpoint_save_path = f"dynamic_rlhf_models/{session_id}_checkpoint_{checkpoint_step}"
        drlhf.save(checkpoint_save_path, checkpoint_step, str(exp_id))
        print(f"Saved full DynamicRLHF checkpoint to: {checkpoint_save_path}")

        # Process trajectories to generate videos, rewards, uncertainty files, and thumbnails
        session["status"] = "processing_trajectories"
        processing_success = await process_dynamic_rlhf_trajectories(
            env_name=env_name, exp_id=str(exp_id), checkpoint_step=checkpoint_step
        )

        if processing_success:
            print("Successfully processed trajectories for visualization")

            # Generate projections and uncertainty maps for visualization
            session["status"] = "generating_projections"
            projection_success = await generate_dynamic_rlhf_projections(
                env_name=env_name, exp_id=str(exp_id), checkpoint_step=checkpoint_step
            )

            if projection_success:
                print("Successfully generated projections and uncertainty maps")
            else:
                print("Warning: Failed to generate projections and uncertainty maps")
        else:
            print("Warning: Failed to process trajectories for visualization")

        # Add the new checkpoint to the experiment's checkpoint list if it's not there
        if checkpoint_step > 0:
            exp: Experiment = await db_handler.get_single_entry(database, Experiment, key=exp_id)
            existing_checkpoints = exp.checkpoint_list if exp.checkpoint_list else []
            if checkpoint_step not in existing_checkpoints:
                existing_checkpoints.append(checkpoint_step)
                existing_checkpoints.sort()
                await db_handler.update_entry(
                    database,
                    Experiment,
                    key=exp_id,
                    data={"checkpoint_list": existing_checkpoints},
                )
                print(f"Added checkpoint {checkpoint_step} to experiment {exp_id}")

        # Mark training as completed
        session["status"] = "completed"
        session["completed_at"] = time.time()
        print(f"Training iteration {phase} completed successfully")

    except Exception as e:
        # Mark training as failed
        session["status"] = "error"
        session["error"] = str(e)
        import traceback

        error_msg = f"Error in background training iteration: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)


@router.post("/train_iteration", response_model=dict[str, Any])
async def train_iteration(request: Request, background_tasks: BackgroundTasks):
    """
    Performs a training iteration using DynamicRLHF with human feedback.

    Args:
        request: The HTTP request

    Returns:
        JSON response with submission status
    """
    # Get session_id, experiment_id, and phase from query parameters
    session_id = request.query_params.get("session_id", None)
    experiment_id = request.query_params.get("experiment_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if not session_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing session_id parameter"})

    if not experiment_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing experiment_id parameter"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        # Start simulated training
        simulation_key = f"{session_id}_{phase}"

        # Get the next training step number (increment from previous phases)
        training_step = 1
        for key, data in simulation_store.items():
            if key.startswith(f"{session_id}_") and "final_training_step" in data:
                training_step = max(training_step, data["final_training_step"] + 1)

        simulation_store[simulation_key] = {
            "status": "training",
            "start_time": time.time(),
            "training_step": training_step,
            "uncertainty": random.uniform(0.1, 0.5),
            "avg_reward": random.uniform(0.3, 0.8),
            "progress": 0.0,
        }

        return JSONResponse(
            content={
                "phaseStatus": "training_started",
                "message": "Training iteration started successfully",
                "phaseTrainingStep": training_step,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
                "session_id": session_id,
                "num_feedback": 0,
            }
        )

    # DynamicRLHF-based training logic
    try:
        # Check if we have an active RLHF session, if not create one
        if session_id not in active_rlhf_sessions:
            print(f"No active session found for {session_id}, initializing new session...")
            # Auto-initialize session if it doesn't exist
            try:
                session_data = await initialize_dynamic_rlhf_session(session_id, experiment_id, 5)
                active_rlhf_sessions[session_id] = session_data
                print(f"Auto-initialized DynamicRLHF session: {session_id}")
            except Exception as init_error:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": f"Failed to initialize DynamicRLHF session: {str(init_error)}"},
                )
        else:
            print(f"Found existing session for {session_id}")

        # Update phase for existing session
        active_rlhf_sessions[session_id]["phase"] = phase

        # Get the session and update training step
        session = active_rlhf_sessions[session_id]
        session["training_step"] = session.get("training_step", 0) + 1
        session["status"] = "starting"

        # Start the background training task
        background_tasks.add_task(
            run_training_iteration_background,
            session_id=session_id,
            experiment_id=experiment_id,
            phase=phase,
            active_rlhf_sessions=active_rlhf_sessions,
        )

        print(f"Started background training task for session {session_id}, phase {phase}")

        return JSONResponse(
            content={
                "phaseStatus": "training_started",
                "message": "DynamicRLHF training iteration started successfully",
                "phaseTrainingStep": session["training_step"],
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
                "session_id": session_id,
                "num_feedback": 0,
            }
        )

    except Exception as e:
        import traceback

        error_msg = f"Error starting DynamicRLHF training iteration: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        return JSONResponse(
            status_code=500,
            content={
                "phaseStatus": "error",
                "message": f"Failed to start training: {str(e)}",
                "phaseTrainingStep": 0,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
            },
        )


@router.get("/get_training_status", response_model=dict[str, Any])
async def get_training_status(request: Request):
    """
    Returns the training status of the current DynamicRLHF training iteration.

    Args:
        request: The HTTP request

    Returns:
        Dictionary with training status
    """
    session_id = request.query_params.get("session_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if session_id is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No session id given"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        simulation_key = f"{session_id}_{phase}"
        if simulation_key in simulation_store:
            sim_data = simulation_store[simulation_key]
            elapsed_time = time.time() - sim_data["start_time"]

            # Calculate progress (complete after 5 seconds)
            progress = min(elapsed_time / 5.0, 1.0)
            sim_data["progress"] = progress
            # Keep the training step constant during the 5-second simulation

            if progress >= 1.0:
                sim_data["status"] = "completed"

            return JSONResponse(
                content={
                    "status": sim_data["status"],
                    "message": ("Training in progress" if sim_data["status"] == "training" else "Training completed"),
                    "progress": progress,
                    "training_step": sim_data["training_step"],
                }
            )
        else:
            return JSONResponse(
                content={
                    "status": "idle",
                    "message": "No training in progress",
                }
            )

    # DynamicRLHF-based training status
    if session_id not in active_rlhf_sessions:
        return JSONResponse(
            content={
                "status": "idle",
                "message": "No DynamicRLHF session active",
            }
        )

    session = active_rlhf_sessions[session_id]

    # Check if we're waiting for human feedback
    feedback_status_file = Path(f"sessions/{session_id}/feedback_status.json")
    if feedback_status_file.exists():
        try:
            with open(feedback_status_file, "r") as f:
                feedback_status = json.load(f)

            if feedback_status.get("status") == "candidates_ready":
                return JSONResponse(
                    content={
                        "status": "waiting_for_feedback",
                        "message": "Waiting for human feedback",
                        "progress": 0.5,  # 50% complete, waiting for feedback
                        "training_step": session.get("training_step", 0),
                    }
                )
            elif feedback_status.get("status") == "feedback_received":
                return JSONResponse(
                    content={
                        "status": "processing_feedback",
                        "message": "Processing received feedback",
                        "progress": 0.8,  # 80% complete, processing feedback
                        "training_step": session.get("training_step", 0),
                    }
                )
        except Exception as e:
            print(f"Error reading feedback status: {e}")

    # Return detailed training status based on background task progress
    status = session.get("status", "idle")

    # Define progress mapping for different stages
    progress_mapping = {
        "idle": 0.0,
        "starting": 0.05,
        "training": 0.1,
        "collecting_initial_data": 0.2,
        "loading_feedback": 0.3,
        "training_reward_models": 0.5,
        "training_rl_agent": 0.7,
        "collecting_trajectories": 0.8,
        "saving_models": 0.85,
        "saving_checkpoint": 0.9,
        "processing_trajectories": 0.95,
        "generating_projections": 0.97,
        "completed": 1.0,
        "error": 0.0,
    }

    # Define human-readable messages for each status
    status_messages = {
        "idle": "No training in progress",
        "starting": "Initializing training iteration...",
        "training": "Training in progress...",
        "collecting_initial_data": "Collecting initial trajectories with untrained models...",
        "loading_feedback": "Loading and processing human feedback...",
        "training_reward_models": "Training reward models with feedback...",
        "training_rl_agent": "Training RL agent with updated reward models...",
        "collecting_trajectories": "Collecting new trajectories with trained agent...",
        "saving_models": "Saving trained models...",
        "saving_checkpoint": "Saving training checkpoint...",
        "processing_trajectories": "Processing trajectories for visualization...",
        "generating_projections": "Generating projections and uncertainty maps...",
        "completed": "Training iteration completed successfully",
        "error": f"Training failed: {session.get('error', 'Unknown error')}",
    }

    progress = progress_mapping.get(status, 0.0)
    message = status_messages.get(status, f"DynamicRLHF session {status}")

    return JSONResponse(
        content={
            "status": status,
            "message": message,
            "progress": progress,
            "training_step": session.get("training_step", 0),
        }
    )


@router.get("/get_training_results", response_model=dict[str, Any])
async def get_training_results(request: Request):
    """
    Returns the training results of the current DynamicRLHF training iteration:
    uncertainty and average predicted reward.

    Args:
        request: The HTTP request

    Returns:
        Dictionary with training results
    """
    session_id = request.query_params.get("session_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if session_id is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No session id given"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        simulation_key = f"{session_id}_{phase}"
        if simulation_key in simulation_store:
            sim_data = simulation_store[simulation_key]
            elapsed_time = time.time() - sim_data["start_time"]

            # Calculate progress (complete after 5 seconds)
            progress = min(elapsed_time / 5.0, 1.0)
            training_complete = progress >= 1.0

            # Update simulation data
            sim_data["progress"] = progress
            # Keep the training step constant during the 5-second simulation

            if training_complete:
                sim_data["status"] = "completed"
                # Store final training step for next phase increment
                final_step_key = f"{session_id}_final"
                simulation_store[final_step_key] = {"final_training_step": sim_data["training_step"]}
                # Clean up simulation data after completion
                del simulation_store[simulation_key]

            return JSONResponse(
                content={
                    "phaseStatus": "completed" if training_complete else "training",
                    "message": ("Training completed successfully" if training_complete else "Training in progress"),
                    "phaseTrainingStep": sim_data["training_step"],
                    "phaseUncertainty": sim_data["uncertainty"],
                    "phaseReward": sim_data["avg_reward"],
                    "trainingComplete": training_complete,
                    "modelPath": "",
                    "metrics": {
                        "training_step": sim_data["training_step"],
                        "progress": progress,
                        "simulated": True,
                    },
                }
            )
        else:
            return JSONResponse(
                content={
                    "phaseStatus": "idle",
                    "message": "No training data available",
                    "phaseTrainingStep": 0,
                    "phaseUncertainty": 0.0,
                    "phaseReward": 0.0,
                    "trainingComplete": False,
                    "modelPath": "",
                    "metrics": {},
                }
            )

    # DynamicRLHF-based training results
    if session_id not in active_rlhf_sessions:
        return JSONResponse(
            content={
                "phaseStatus": "idle",
                "message": "No DynamicRLHF session active",
                "phaseTrainingStep": 0,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
                "trainingComplete": False,
                "modelPath": "",
                "metrics": {},
            }
        )

    session = active_rlhf_sessions[session_id]
    drlhf = session["drlhf"]

    # Check if we have completed feedback collection and training for this phase
    feedback_status_file = Path(f"sessions/{session_id}/feedback_status.json")
    training_complete = False
    uncertainty = 0.0
    avg_reward = 0.0

    if feedback_status_file.exists():
        try:
            with open(feedback_status_file, "r") as f:
                feedback_status = json.load(f)

            # Extract actual predicted rewards and uncertainties from the feedback status
            uncertainty = feedback_status.get("avg_uncertainty", 0.0)
            avg_reward = feedback_status.get("avg_predicted_reward", 0.0)

            # If we have processed feedback, consider training complete for this phase
            if feedback_status.get("status") == "feedback_received":
                training_complete = True
            elif feedback_status.get("status") == "candidates_ready":
                # Training iteration has been completed, candidates are ready for feedback
                training_complete = True
        except Exception as e:
            print(f"Error reading feedback status: {e}")

    # Get basic metrics from the session
    metrics = {
        "training_step": session.get("training_step", 0),
        "feedback_types": session.get("feedback_types", []),
        "env_name": session.get("env_name", ""),
        "algorithm": session.get("algorithm", ""),
        "dynamic_rlhf": True,
    }

    return JSONResponse(
        content={
            "phaseStatus": "completed" if training_complete else "training",
            "message": ("DynamicRLHF training completed" if training_complete else "DynamicRLHF training in progress"),
            "phaseTrainingStep": session.get("training_step", 0),
            "phaseUncertainty": uncertainty,
            "phaseReward": avg_reward,
            "trainingComplete": training_complete,
            "modelPath": "",  # Would be populated with actual model path
            "metrics": metrics,
        }
    )

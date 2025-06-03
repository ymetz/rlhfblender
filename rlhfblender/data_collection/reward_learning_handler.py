"""Module for handling reward model training in an interactive RLHF process."""

import json
import logging
import os
import pickle
import signal
import subprocess
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Define training status enums
class TrainingStatus(str, Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PROCESSING_FEEDBACK = "processing_feedback"
    TRAINING = "training"
    COLLECTING_EPISODES = "collecting_episodes"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingResult(BaseModel):
    """Model for storing training results."""

    uncertainty: float = 0.0
    avg_predicted_reward: float = 0.0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    training_complete: bool = False
    trained_model_path: str = ""
    new_episodes: List[Dict] = []
    metrics: Dict[str, Any] = {}


class RewardModelHandler:
    """
    Handler for managing reward model training and episode collection.

    This class is responsible for:
    1. Initializing training processes
    2. Processing feedback
    3. Training reward models
    4. Collecting new episodes with updated models
    5. Providing status updates
    """

    def __init__(self, exp: Optional[Any] = None, session_id: Optional[str] = None):
        """
        Initialize the reward model handler.

        Args:
            exp: The experiment configuration
            session_id: Unique identifier for the training session
        """
        self.exp = exp
        self.session_id = session_id or str(uuid.uuid4())
        self.status = TrainingStatus.NOT_STARTED
        self.process = None
        self.result = TrainingResult()

        # Create session directory
        self.session_dir = Path(f"sessions/{self.session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(f"reward_handler_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.session_dir / "training.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)

        # Status file path
        self.status_file = self.session_dir / "status.json"
        self.result_file = self.session_dir / "result.json"

        # Initialize status file
        self._update_status(TrainingStatus.NOT_STARTED)

        # Save experiment info if provided
        if exp:
            with open(self.session_dir / "experiment.json", "w") as f:
                # Convert to dict for serialization
                exp_dict = {k: v for k, v in exp.__dict__.items() if not k.startswith("_")}
                json.dump(exp_dict, f)

    def _update_status(self, status: TrainingStatus, message: str = ""):
        """Update the status file with the current training status."""
        self.status = status
        status_data = {"status": status, "message": message, "timestamp": time.time()}
        with open(self.status_file, "w") as f:
            json.dump(status_data, f)
        self.logger.info(f"Status updated to {status}: {message}")

    def _update_result(self, result: dict):
        """Update the result file with the current training results."""
        # Update the result object
        for key, value in result.items():
            if hasattr(self.result, key):
                setattr(self.result, key, value)

        # Save to file
        with open(self.result_file, "w") as f:
            json.dump(self.result.dict(), f)

        self.logger.info(f"Updated training results: {result}")

    def submit_feedback(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Submit feedback for training and start the training process.

        Args:
            feedback: List of feedback instances

        Returns:
            Dictionary with submission status
        """
        try:
            # Save feedback to file
            feedback_file = self.session_dir / "feedback.pkl"
            with open(feedback_file, "wb") as f:
                pickle.dump(feedback, f)

            self.logger.info(f"Saved {len(feedback)} feedback items to {feedback_file}")

            # Start training process
            self._start_training_process(feedback_file)

            return {"status": "success", "message": "Feedback submitted and training started", "session_id": self.session_id}

        except Exception as e:
            self.logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, message=str(e))
            return {"status": "error", "message": f"Failed to submit feedback: {str(e)}", "session_id": self.session_id}

    def _start_training_process(self, feedback_file: Path):
        """
        Start the training process in a separate subprocess.

        Args:
            feedback_file: Path to the saved feedback file
        """
        self._update_status(TrainingStatus.INITIALIZING)

        # Create a script to run the training process
        script_file = self.session_dir / "training_script.py"

        # Create command to run the script
        cmd = [
            "python",
            str(script_file),
            "--feedback-file",
            str(feedback_file),
            "--session-dir",
            str(self.session_dir),
            "--env",
            self.exp.env_id if self.exp else "HalfCheetah-v5",
            "--algorithm",
            self.exp.algorithm if self.exp else "sac",
            "--seed",
            str(self.exp.seed if self.exp else 0),
        ]

        # Start process
        self.logger.info(f"Starting training process with command: {' '.join(cmd)}")

        # Use subprocess instead of multiprocessing for better isolation
        # Redirect output to log file
        log_file = open(self.session_dir / "process.log", "w")
        self.process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True  # Create a new process group
        )

        self.logger.info(f"Started training process with PID {self.process.pid}")

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get the current status of the training process.

        Returns:
            Dictionary with status information
        """
        if not self.status_file.exists():
            return {"status": TrainingStatus.NOT_STARTED, "message": "Training not started", "timestamp": time.time()}

        try:
            with open(self.status_file, "r") as f:
                status_data = json.load(f)

            # Check if process is still running
            if self.process:
                status_data["process_running"] = self.process.poll() is None
                if not status_data["process_running"] and status_data["status"] not in [
                    TrainingStatus.COMPLETED,
                    TrainingStatus.FAILED,
                ]:
                    # Process terminated but not marked as completed or failed - something went wrong
                    status_data["status"] = TrainingStatus.FAILED
                    status_data["message"] = "Training process terminated unexpectedly"

            return status_data

        except Exception as e:
            self.logger.error(f"Error getting training status: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to get training status: {str(e)}", "timestamp": time.time()}

    def get_training_results(self) -> Dict[str, Any]:
        """
        Get the results of the training process.

        Returns:
            Dictionary with training results
        """
        if not self.result_file.exists():
            return self.result.dict()

        try:
            with open(self.result_file, "r") as f:
                result_data = json.load(f)

            return result_data

        except Exception as e:
            self.logger.error(f"Error getting training results: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to get training results: {str(e)}",
                "timestamp": time.time(),
                **self.result.dict(),
            }

    def cancel_training(self) -> Dict[str, str]:
        """
        Cancel the running training process.

        Returns:
            Dictionary with cancellation status
        """
        if not self.process or self.process.poll() is not None:
            return {"status": "warning", "message": "No active training process to cancel"}

        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            # Wait a bit for graceful termination
            time.sleep(2)

            # Force kill if still running
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

            self._update_status(TrainingStatus.FAILED, message="Training cancelled by user")

            return {"status": "success", "message": "Training process cancelled"}

        except Exception as e:
            self.logger.error(f"Error cancelling training: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to cancel training: {str(e)}"}

    def initial_episode_collection(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run initial episode collection with a random policy.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            Dictionary with collected episodes
        """
        self._update_status(TrainingStatus.COLLECTING_EPISODES, "Collecting initial episodes")

        try:
            # Run the episode collection script as a subprocess
            cmd = [
                "python",
                "collect_episodes.py",
                "--session-dir",
                str(self.session_dir),
                "--env",
                self.exp.env_id if self.exp else "HalfCheetah-v5",
                "--num-episodes",
                str(num_episodes),
                "--random-policy",
                "true",  # Use random policy for initial collection
            ]

            # Run process and wait for completion
            log_file = open(self.session_dir / "collection.log", "w")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            process.wait()

            # Load episodes from file
            episodes_file = self.session_dir / "episodes.json"
            if episodes_file.exists():
                with open(episodes_file, "r") as f:
                    episodes = json.load(f)

                self._update_status(TrainingStatus.NOT_STARTED, f"Collected {len(episodes)} initial episodes")

                return {"status": "success", "message": f"Collected {len(episodes)} initial episodes", "episodes": episodes}
            else:
                self._update_status(TrainingStatus.FAILED, "Failed to collect initial episodes")
                return {"status": "error", "message": "Failed to collect initial episodes", "episodes": []}

        except Exception as e:
            self.logger.error(f"Error collecting initial episodes: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, str(e))
            return {"status": "error", "message": f"Failed to collect initial episodes: {str(e)}", "episodes": []}

    def evaluate_reward_model(self, model_path: str, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate a trained reward model on episodes.

        Args:
            model_path: Path to the trained reward model
            num_episodes: Number of episodes to evaluate on

        Returns:
            Dictionary with evaluation results
        """
        self._update_status(TrainingStatus.EVALUATING, f"Evaluating model {model_path}")

        try:
            # Run the evaluation script
            cmd = [
                "python",
                "evaluate_model.py",
                "--session-dir",
                str(self.session_dir),
                "--env",
                self.exp.env_id if self.exp else "HalfCheetah-v5",
                "--model-path",
                model_path,
                "--num-episodes",
                str(num_episodes),
            ]

            # Run process and wait for completion
            log_file = open(self.session_dir / "evaluation.log", "w")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            process.wait()

            # Load evaluation results
            eval_file = self.session_dir / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    results = json.load(f)

                self._update_status(TrainingStatus.NOT_STARTED, f"Evaluated model on {num_episodes} episodes")

                # Update result file with evaluation metrics
                self._update_result(
                    {
                        "uncertainty": results.get("uncertainty", 0.0),
                        "avg_predicted_reward": results.get("avg_reward", 0.0),
                        "metrics": {**self.result.metrics, **results.get("metrics", {})},
                    }
                )

                return {"status": "success", "message": f"Evaluated model on {num_episodes} episodes", **results}
            else:
                self._update_status(TrainingStatus.FAILED, "Failed to evaluate model")
                return {"status": "error", "message": "Failed to evaluate model", "uncertainty": 0.0, "avg_reward": 0.0}

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, str(e))
            return {"status": "error", "message": f"Failed to evaluate model: {str(e)}", "uncertainty": 0.0, "avg_reward": 0.0}

    def collect_episodes_with_model(self, model_path: str, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Collect episodes using a trained reward model.

        Args:
            model_path: Path to the trained reward model
            num_episodes: Number of episodes to collect

        Returns:
            Dictionary with collected episodes
        """
        self._update_status(TrainingStatus.COLLECTING_EPISODES, f"Collecting episodes with model {model_path}")

        try:
            # Run the episode collection script with the model
            cmd = [
                "python",
                "collect_episodes.py",
                "--session-dir",
                str(self.session_dir),
                "--env",
                self.exp.env_id if self.exp else "HalfCheetah-v5",
                "--num-episodes",
                str(num_episodes),
                "--random-policy",
                "false",  # Use the reward model
                "--reward-model-path",
                model_path,
            ]

            # Run process and wait for completion
            log_file = open(self.session_dir / "model_collection.log", "w")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            process.wait()

            # Load episodes from file
            episodes_file = self.session_dir / "episodes.json"
            if episodes_file.exists():
                with open(episodes_file, "r") as f:
                    episodes = json.load(f)

                self._update_status(TrainingStatus.NOT_STARTED, f"Collected {len(episodes)} episodes with model")

                # Update result with new episodes
                self._update_result({"new_episodes": episodes})

                return {"status": "success", "message": f"Collected {len(episodes)} episodes with model", "episodes": episodes}
            else:
                self._update_status(TrainingStatus.FAILED, "Failed to collect episodes with model")
                return {"status": "error", "message": "Failed to collect episodes with model", "episodes": []}

        except Exception as e:
            self.logger.error(f"Error collecting episodes with model: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, str(e))
            return {"status": "error", "message": f"Failed to collect episodes with model: {str(e)}", "episodes": []}

    def predict_uncertainty_for_states(self, states: list[list[float]], model_path: str) -> dict[str, Any]:
        """
        Predict uncertainty for given states using a trained reward model.

        Args:
            states: List of states to predict uncertainty for
            model_path: Path to the trained reward model

        Returns:
            Dictionary with predicted uncertainties
        """
        self._update_status(TrainingStatus.EVALUATING, f"Predicting uncertainty for {len(states)} states")

        try:
            # Run the prediction script
            cmd = [
                "python",
                "predict_uncertainty.py",
                "--session-dir",
                str(self.session_dir),
                "--model-path",
                model_path,
                "--states-file",
                str(self.session_dir / "states.json"),
            ]

            # Save states to file
            with open(self.session_dir / "states.json", "w") as f:
                json.dump(states, f)

            # Run process and wait for completion
            log_file = open(self.session_dir / "prediction.log", "w")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            process.wait()

            # Load predictions from file
            predictions_file = self.session_dir / "predictions.json"
            if predictions_file.exists():
                with open(predictions_file, "r") as f:
                    predictions = json.load(f)

                self._update_status(TrainingStatus.NOT_STARTED, f"Predicted uncertainties for {len(states)} states")

                return {
                    "status": "success",
                    "message": f"Predicted uncertainties for {len(states)} states",
                    "predictions": predictions,
                }
            else:
                self._update_status(TrainingStatus.FAILED, "Failed to predict uncertainties")
                return {"status": "error", "message": "Failed to predict uncertainties", "predictions": []}

        except Exception as e:
            self.logger.error(f"Error predicting uncertainties: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, str(e))
            return {"status": "error", "message": f"Failed to predict uncertainties: {str(e)}", "predictions": []}

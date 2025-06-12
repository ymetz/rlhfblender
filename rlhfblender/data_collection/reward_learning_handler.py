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

from rlhfblender.data_collection.feedback_translator import FeedbackTranslator


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

    def __init__(self, exp: Optional[Any] = None, session_id: Optional[str] = None, phase: Optional[int] = None):
        """
        Initialize the reward model handler.

        Args:
            exp: The experiment configuration
            session_id: Unique identifier for the training session
            phase: Current training phase/iteration number
        """
        self.exp = exp
        self.session_id = session_id or str(uuid.uuid4())
        self.phase = phase or 0
        self.status = TrainingStatus.NOT_STARTED
        self.process = None
        self.result = TrainingResult()

        # Create phase-specific session directory
        # This ensures each training iteration has its own directory
        base_session_dir = Path(f"sessions/{self.session_id}")
        self.session_dir = base_session_dir / f"phase_{self.phase}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create base session directory for shared files
        self.base_session_dir = base_session_dir
        self.base_session_dir.mkdir(parents=True, exist_ok=True)

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
            json.dump(self.result.model_dump(), f)

        self.logger.info(f"Updated training results: {result}")

    def submit_feedback(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Submit feedback for training and start the training process.

        Args:
            feedback: List of already processed StandardizedFeedback instances

        Returns:
            Dictionary with submission status
        """
        try:
            # Create preprocessed unified dataset from processed feedback
            self._preprocess_feedback_datasets(feedback)

            # Save processed feedback to file for training script
            feedback_file = self.session_dir / "feedback.pkl"
            with open(feedback_file, "wb") as f:
                pickle.dump(feedback, f)

            self.logger.info(f"Processed and saved {len(feedback)} feedback items to {feedback_file}")

            # Start training process with preprocessed datasets
            self._start_training_process(feedback_file)

            return {"status": "success", "message": "Feedback submitted and training started", "session_id": self.session_id}

        except Exception as e:
            self.logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
            self._update_status(TrainingStatus.FAILED, message=str(e))
            return {"status": "error", "message": f"Failed to submit feedback: {str(e)}", "session_id": self.session_id}

    def _preprocess_feedback_datasets(self, feedback: List[Dict]) -> None:
        """
        Create preprocessed unified datasets from already processed feedback.
        
        Args:
            feedback: List of processed StandardizedFeedback objects (as dicts)
        """
        try:
            self._update_status(TrainingStatus.PROCESSING_FEEDBACK, "Creating unified dataset from processed feedback")
            
            # Convert dict feedback back to StandardizedFeedback objects if needed
            from rlhfblender.data_models.feedback_models import StandardizedFeedback, SimplifiedFeedbackType
            
            standardized_feedbacks = []
            for fb_dict in feedback:
                try:
                    # If it's already a StandardizedFeedback object, use it directly
                    if isinstance(fb_dict, StandardizedFeedback):
                        feedback_obj = fb_dict
                    else:
                        # If it's a dict, try to convert it
                        feedback_obj = StandardizedFeedback(**fb_dict)
                    
                    # Filter out meta feedback - it's only for logging, not training
                    if feedback_obj.feedback_type != SimplifiedFeedbackType.meta:
                        standardized_feedbacks.append(feedback_obj)
                    else:
                        self.logger.info(f"Skipping meta feedback for training: {feedback_obj.content}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to parse feedback item: {e}")
                    continue
            
            # Get environment name from experiment
            env_name = self.exp.env_id if self.exp and hasattr(self.exp, 'env_id') else ""
            
            # Organize feedback by type for individual model training
            feedbacks_by_type = {}
            for fb in standardized_feedbacks:
                fb_type = fb.feedback_type.value
                if fb_type not in feedbacks_by_type:
                    feedbacks_by_type[fb_type] = []
                feedbacks_by_type[fb_type].append(fb)
            
            # Save one dataset per feedback type for individual training
            for fb_type, type_feedbacks in feedbacks_by_type.items():
                # Create dataset for this specific feedback type
                type_dataset = FeedbackTranslator.create_unified_dataset_from_processed(
                    processed_feedback=type_feedbacks,
                    n_feedback=-1,
                    env_name=env_name
                )
                
                # Save preprocessed dataset for this type
                type_file = self.session_dir / f"{fb_type}_dataset.pkl"
                with open(type_file, "wb") as f:
                    pickle.dump(type_dataset, f)
                
                self.logger.info(f"Saved {len(type_feedbacks)} {fb_type} feedback items as dataset to {type_file}")
                
            self.logger.info(f"Preprocessed {len(standardized_feedbacks)} feedback items into {len(feedbacks_by_type)} type-specific datasets")
            self.logger.info(f"Feedback types found: {list(feedbacks_by_type.keys())}")
            
            if len(standardized_feedbacks) == 0:
                self.logger.warning("No non-meta feedback available for training!")
            
        except Exception as e:
            self.logger.error(f"Error preprocessing feedback datasets: {str(e)}", exc_info=True)
            raise

    def _start_training_process(self, feedback_file: Path):
        """
        Start the training process in a separate subprocess.

        Args:
            feedback_file: Path to the saved feedback file
        """
        self._update_status(TrainingStatus.INITIALIZING)

        # Use the existing training script in the project
        script_file = Path(__file__).parent / "training_script.py"
        
        if not script_file.exists():
            raise FileNotFoundError(f"Training script not found at {script_file}")
        
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
            return self.result.model_dump()

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
                **self.result.model_dump(),
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
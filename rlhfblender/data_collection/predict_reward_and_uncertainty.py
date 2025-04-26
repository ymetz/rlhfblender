"""
Predict uncertainty for a given list of states

To achieve this:
(1) Load reward model and policy from given checkpoint
(2) For each state in list, predict action with policy
(3) For each state, predict reward with reward model
(4) Save predictions to session_dir
"""

import argparse
import json
import os
import time as import_time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# For loading reward models
from multi_type_feedback.networks import SingleCnnNetwork, SingleNetwork
from train_baselines.utils import ALGOS

# For loading policies/agents
from rlhfblender.data_models.agent import BaseAgent


class RewardUncertaintyPredictor:
    """Class for predicting rewards and uncertainty with a reward model"""

    def __init__(
        self,
        reward_model_path: str,
        policy_model_path: Optional[str] = None,
        policy_algorithm: Optional[str] = "ppo",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.reward_model_path = reward_model_path
        self.policy_model_path = policy_model_path
        self.device = device
        self.reward_model = None
        self.policy_model = None
        self.policy_algorithm = policy_algorithm

        # Load reward model
        self._load_reward_model()

        # Load policy model if provided
        if policy_model_path:
            self._load_policy_model()

    def _load_reward_model(self):
        """Load the reward model from checkpoint"""
        print(f"Loading reward model from {self.reward_model_path}")

        try:
            # First try loading as CNN network
            self.reward_model = SingleCnnNetwork.load_from_checkpoint(self.reward_model_path, map_location=self.device)
        except Exception as e:
            print(f"Failed to load as CNN network: {e}")
            # Try loading as regular network
            self.reward_model = SingleNetwork.load_from_checkpoint(self.reward_model_path, map_location=self.device)

        self.reward_model.eval()
        print(f"Successfully loaded reward model")

    def _load_policy_model(self):
        """Load the policy model from checkpoint"""
        print(f"Loading policy model from {self.policy_model_path}")

        self.policy_model = ALGOS[self.policy_algorithm].load(
            self.policy_model_path,
            env=None,
            device=self.device,
        )

        print("Successfully loaded policy model")

    def predict_actions(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict actions for given observations using the policy model

        Args:
            observations: Array of observations [batch_size, *obs_shape]

        Returns:
            Array of predicted actions [batch_size, *action_shape]
        """
        if self.policy_model is None:
            raise ValueError("Policy model not loaded. Cannot predict actions.")

        # Convert observations to appropriate format
        if isinstance(observations, np.ndarray):
            if len(observations.shape) == 3:  # Single observation with channels
                observations = np.expand_dims(observations, axis=0)
            elif len(observations.shape) == 1:  # Single vector observation
                observations = np.expand_dims(observations, axis=0)

        actions, _ = self.policy_model.predict(observations, deterministic=True)
        return actions

    def _one_hot_encode_actions(self, actions: np.ndarray, n_actions: int) -> torch.Tensor:
        """
        Convert discrete actions to one-hot encoding.

        Args:
            actions: Numpy array of discrete actions
            n_actions: Number of possible actions

        Returns:
            torch.Tensor: One-hot encoded actions
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long()
        else:
            actions = torch.tensor(actions, dtype=torch.long)

        # Handle single action case
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)

        one_hot = torch.zeros(actions.shape[0], n_actions, device=self.device)
        one_hot.scatter_(1, actions.unsqueeze(1), 1)
        return one_hot

    def predict_rewards_and_uncertainty(
        self, observations: np.ndarray, actions: np.ndarray, action_space_size: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict rewards and uncertainty for given observations and actions

        Args:
            observations: Array of observations [batch_size, *obs_shape]
            actions: Array of actions [batch_size, *action_shape]
            action_space_size: Size of discrete action space (if applicable)

        Returns:
            Tuple of (predicted_rewards, uncertainties) as numpy arrays
        """
        if self.reward_model is None:
            raise ValueError("Reward model not loaded. Cannot predict rewards.")

        with torch.no_grad():
            # Convert to tensors and move to device
            obs_tensor = torch.as_tensor(observations, device=self.device, dtype=torch.float)

            # Handle actions based on their type (discrete vs continuous)
            if len(actions.shape) == 1 or (len(actions.shape) == 2 and actions.shape[1] == 1):
                # Discrete actions
                if action_space_size is None:
                    # Estimate action space size from actions
                    action_space_size = int(np.max(actions)) + 1
                action_tensor = self._one_hot_encode_actions(actions, action_space_size)
            else:
                # Continuous actions
                action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float)

            # Reshape for reward model if needed based on model's expected input format
            batch_size = obs_tensor.shape[0]

            # Add sequence dimension needed by reward model (batch_size, seq_len, ...)
            if len(obs_tensor.shape) == 2:  # (batch_size, obs_dim)
                obs_tensor = obs_tensor.unsqueeze(1)  # Add sequence dim
            elif len(obs_tensor.shape) > 2 and obs_tensor.shape[1] != 1:  # Reshape if needed
                # For CNN inputs, likely (batch_size, channels, height, width)
                # Rewire to (batch_size, 1, channels, height, width)
                obs_tensor = obs_tensor.unsqueeze(1)

            # Add sequence dimension to actions if needed
            if len(action_tensor.shape) == 2:  # (batch_size, action_dim)
                action_tensor = action_tensor.unsqueeze(1)

            # Get predictions
            if hasattr(self.reward_model, "ensemble_count") and self.reward_model.ensemble_count > 1:

                # determine shape for tiling: ensemble count, then ones for following dimensions
                obs_tile_shape = (self.reward_model.ensemble_count,) + (1,) * (len(obs_tensor.shape) - 1)
                obs_tensor = torch.tile(obs_tensor, obs_tile_shape)

                action_tile_shape = (self.reward_model.ensemble_count,) + (1,) * (len(action_tensor.shape) - 1)
                action_tensor = torch.tile(action_tensor, action_tile_shape)

                predictions = self.reward_model(obs_tensor, action_tensor)

                # Reshape predictions to get ensemble rewards per sample
                # Shape: (ensemble_count, batch_size, 1)
                predictions = predictions.view(self.reward_model.ensemble_count, batch_size, -1)

                # Calculate mean and std across ensemble models
                mean_rewards = torch.mean(predictions, dim=0).squeeze(-1)
                uncertainties = torch.std(predictions, dim=0).squeeze(-1)
            else:
                # For single models
                predictions = self.reward_model(obs_tensor, action_tensor)
                mean_rewards = predictions.squeeze(-1)
                # No real uncertainty for non-ensemble models, return zeros
                uncertainties = torch.zeros_like(mean_rewards)

            return mean_rewards.cpu().numpy(), uncertainties.cpu().numpy()

    def predict_for_states(
        self, observations: np.ndarray, actions: Optional[np.ndarray] = None, action_space_size: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict rewards and uncertainty for given states

        Args:
            observations: Array of observations [batch_size, *obs_shape]
            actions: Optional array of actions [batch_size, *action_shape]
                     If None, actions will be predicted using the policy model
            action_space_size: Size of discrete action space (if applicable)

        Returns:
            Dictionary with predicted rewards, uncertainty, and actions (if predicted)
        """
        if actions is None:
            if self.policy_model is None:
                raise ValueError("Policy model not loaded and no actions provided.")
            actions = self.predict_actions(observations)

        rewards, uncertainty = self.predict_rewards_and_uncertainty(observations, actions, action_space_size)

        return {"rewards": rewards, "uncertainty": uncertainty, "actions": actions}

    def predict_for_inverse_projections(
        self,
        inverse_projection_file: str,
        original_obs: np.ndarray,
        original_actions: np.ndarray,
        action_space_size: int = None,
    ) -> Dict[str, any]:
        """
        Load inverse projections and predict rewards and uncertainty
        for both original data points and grid reconstructions

        Args:
            inverse_projection_file: Path to the inverse projection file
            action_space_size: Size of discrete action space (if applicable)

        Returns:
            Dictionary with predictions for both original data and grid data
        """
        # Load inverse projection data
        print(f"Loading inverse projection data from {inverse_projection_file}")
        with open(inverse_projection_file, "r") as f:
            projection_data = json.load(f)

        if "inverse_results" not in projection_data or not projection_data["inverse_results"]:
            raise ValueError("No inverse projection results found in file.")

        # Extract grid data
        grid_samples = projection_data["inverse_results"]["grid_samples"]
        if "reconstructions" not in grid_samples:
            raise ValueError("No reconstructions found in inverse projection results.")

        # Extract original data from the projection data
        original_coordinates = np.array(projection_data.get("projection", []))

        # Extract grid reconstructions
        grid_reconstructions = np.array(grid_samples["reconstructions"])
        grid_coordinates = np.array(grid_samples["coords"])
        grid_x = np.array(grid_samples["grid_x"])
        grid_y = np.array(grid_samples["grid_y"])

        results = {
            "grid_coordinates": grid_coordinates.tolist(),
            "grid_x": grid_x.tolist(),
            "grid_y": grid_y.tolist(),
        }

        # Predict for original data if available
        if len(original_obs) > 0:
            print(f"Predicting rewards and uncertainty for {len(original_obs)} original data points from inverse projection")
            # If actions are available, use them; otherwise predict
            if len(original_actions) == len(original_obs):
                original_rewards, original_uncertainties = self.predict_rewards_and_uncertainty(
                    original_obs, original_actions, action_space_size
                )
            else:
                if self.policy_model is not None:
                    original_actions = self.predict_actions(original_obs)
                    original_rewards, original_uncertainties = self.predict_rewards_and_uncertainty(
                        original_obs, original_actions, action_space_size
                    )
                else:
                    print("No policy model available and no actions provided for original data. Skipping predictions.")
                    original_rewards, original_uncertainties = np.array([]), np.array([])

            results["original_coordinates"] = original_coordinates.tolist()
            results["original_predictions"] = original_rewards.tolist()
            results["original_uncertainties"] = original_uncertainties.tolist()
            results["original_actions"] = (
                original_actions.tolist() if isinstance(original_actions, np.ndarray) else original_actions
            )
        else:
            print("No original observations found in the projection data.")
            results["original_coordinates"] = []
            results["original_predictions"] = []
            results["original_uncertainties"] = []
            results["original_actions"] = []

        # Predict for grid data
        print(f"Predicting rewards and uncertainty for {len(grid_reconstructions)} grid points")
        if self.policy_model is not None:
            # Predict actions for each reconstructed observation
            grid_actions = self.predict_actions(grid_reconstructions)
        else:
            # Use random actions if no policy model is available
            if action_space_size is None:
                # Assume a default action space size
                action_space_size = 4

            if len(grid_reconstructions.shape) > 3:  # Image observations
                # Probably a discrete action space
                grid_actions = np.random.randint(0, action_space_size, size=grid_reconstructions.shape[0])
            else:
                # Probably a continuous action space with dim=1
                grid_actions = np.random.uniform(-1, 1, size=(grid_reconstructions.shape[0], 1))

        grid_rewards, grid_uncertainties = self.predict_rewards_and_uncertainty(
            grid_reconstructions, grid_actions, action_space_size
        )

        results["grid_predictions"] = grid_rewards.tolist()
        results["grid_uncertainties"] = grid_uncertainties.tolist()
        results["grid_actions"] = grid_actions.tolist() if isinstance(grid_actions, np.ndarray) else grid_actions

        # Include file info for reference
        results["source_file"] = os.path.basename(inverse_projection_file)
        results["prediction_timestamp"] = import_time.strftime("%Y-%m-%d %H:%M:%S")

        print("RESULTS", results.keys(), results["original_coordinates"])

        return results

    def save_predictions(
        self, predictions: Dict[str, any], output_dir: str, filename_prefix: str = "predictions", source_file: str = None
    ) -> str:
        """
        Save predictions to output directory in JSON format

        Args:
            predictions: Dictionary with prediction results
            output_dir: Directory to save predictions
            filename_prefix: Prefix for output files
            source_file: Original source file path (for inverse projections)

        Returns:
            Path to saved predictions file
        """
        os.makedirs(output_dir, exist_ok=True)

        # For inverse projections, use the source filename with an appendix
        if source_file and "source_file" in predictions:
            base_name = os.path.splitext(predictions["source_file"])[0]
            output_json_path = os.path.join(output_dir, f"{base_name}_inverse_predictions.json")
        else:
            output_json_path = os.path.join(output_dir, f"{filename_prefix}.json")

        # Create a JSON-serializable version of the predictions
        json_predictions = {}

        # Handle different prediction types
        if "grid_predictions" in predictions:
            # This is an inverse projection result with grid and possibly original data
            json_predictions = predictions  # Already JSON-ready from predict_for_inverse_projections

            # Add some summary statistics
            summary = {
                "grid_rewards_mean": float(np.mean(predictions["grid_predictions"])),
                "grid_rewards_std": float(np.std(predictions["grid_predictions"])),
                "grid_rewards_min": float(np.min(predictions["grid_predictions"])),
                "grid_rewards_max": float(np.max(predictions["grid_predictions"])),
                "grid_uncertainty_mean": float(np.mean(predictions["grid_uncertainties"])),
                "grid_uncertainty_std": float(np.std(predictions["grid_uncertainties"])),
            }

            if predictions["original_predictions"] and len(predictions["original_predictions"]) > 0:
                summary.update(
                    {
                        "original_rewards_mean": float(np.mean(predictions["original_predictions"])),
                        "original_rewards_std": float(np.std(predictions["original_predictions"])),
                        "original_rewards_min": float(np.min(predictions["original_predictions"])),
                        "original_rewards_max": float(np.max(predictions["original_predictions"])),
                        "original_uncertainty_mean": float(np.mean(predictions["original_uncertainties"])),
                        "original_uncertainty_std": float(np.std(predictions["original_uncertainties"])),
                    }
                )

            json_predictions["summary"] = summary
        else:
            # This is a regular episode prediction
            # Convert numpy arrays to lists
            for key, value in predictions.items():
                if isinstance(value, np.ndarray):
                    json_predictions[key] = value.tolist()
                else:
                    json_predictions[key] = value

            # Add summary statistics
            if "rewards" in predictions:
                summary = {
                    "reward_mean": float(np.mean(predictions["rewards"])),
                    "reward_std": float(np.std(predictions["rewards"])),
                    "reward_min": float(np.min(predictions["rewards"])),
                    "reward_max": float(np.max(predictions["rewards"])),
                    "uncertainty_mean": float(np.mean(predictions["uncertainty"])) if "uncertainty" in predictions else 0,
                    "uncertainty_std": float(np.std(predictions["uncertainty"])) if "uncertainty" in predictions else 0,
                    "num_samples": len(predictions["rewards"]),
                }
                json_predictions["summary"] = summary

        # Save JSON file
        with open(output_json_path, "w") as f:
            json.dump(json_predictions, f, indent=2)

        print(f"Saved predictions to {output_json_path}")

        return output_json_path


def load_episodes(episode_path: str) -> Dict[str, np.ndarray]:
    """
    Load episode data from an NPZ file

    Args:
        episode_path: Path to the episode NPZ file

    Returns:
        Dictionary with episode data
    """
    print(f"Loading episode data from {episode_path}")

    # either episode_path is a directory or a file:
    # if it is a file load it directly
    # if it is a directory, the episodes are saved as benchmark_<episode_num>.npz (in this case load all in order and concatenate)
    if os.path.isdir(episode_path):
        # Get all NPZ files in the directory
        episode_files = sorted(Path(episode_path).glob("benchmark_*.npz"))
        if not episode_files:
            raise ValueError(f"No NPZ files found in directory: {episode_path}")

        # Load all episodes and concatenate
        episode_data = []
        for file in episode_files:
            data = np.load(file, allow_pickle=True)
            episode_data.append(data)

        # Concatenate all loaded data, be aware that the npz has multiple keys and arrays
        # We will concatenate the arrays for each key
        concatenated_data = {}
        for key in episode_data[0].files:
            concatenated_data[key] = np.concatenate([data[key] for data in episode_data], axis=0)
        episode_data = concatenated_data
    elif os.path.isfile(episode_path):
        data = np.load(episode_path, allow_pickle=True)

        # Convert to dict to make it easier to work with
        episode_data = {}
        for key in data.files:
            episode_data[key] = data[key]

    return episode_data


def process_env_name(env_name: str) -> str:
    """Process environment name to be compatible with filesystem."""
    return env_name.replace("/", "_").replace(":", "_")


def get_episode_file_path(env_name: str, benchmark_id: int, checkpoint_step: int, episode_num: int) -> str:
    """
    Generate file path for a specific episode

    Args:
        env_name: Environment name
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        episode_num: Episode number

    Returns:
        Path to the episode file
    """
    base_dir = os.path.join(
        "data", "episodes", process_env_name(env_name), f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}"
    )

    return os.path.join(base_dir, f"benchmark_{episode_num}.npz")


def main():
    parser = argparse.ArgumentParser(description="Predict rewards and uncertainty for states")

    # Required arguments
    parser.add_argument("--reward-model", type=str, required=True, help="Path to reward model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save predictions")

    # Optional arguments
    parser.add_argument("--policy-model", type=str, default=None, help="Path to policy model (if predicting actions)")
    parser.add_argument("--policy-algorithm", type=str, default="ppo", help="Algorithm used for the policy model")
    parser.add_argument("--action-space-size", type=int, default=None, help="Size of discrete action space (if applicable)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for predictions"
    )

    # Input data sources - no longer mutually exclusive
    parser.add_argument("--episode-path", type=str, help="Path to episode NPZ file")
    parser.add_argument("--projection-data", type=str, help="Path to projection (+inverse projection) JSON file")

    # Episode identification (required if --env-name is used)
    parser.add_argument("--env-name", type=str, help="Environment name")
    parser.add_argument("--benchmark-id", type=int, help="Benchmark ID")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step")
    parser.add_argument("--episode-num", type=int, help="Episode number")

    args = parser.parse_args()

    # Create predictor
    predictor = RewardUncertaintyPredictor(
        reward_model_path=args.reward_model,
        policy_model_path=args.policy_model,
        policy_algorithm=args.policy_algorithm,
        device=args.device,
    )

    # Check if we need to process both inverse projection and episode data
    if args.projection_data and (
        args.episode_path or (args.env_name and args.benchmark_id and args.checkpoint and args.episode_num)
    ):

        if args.episode_path:
            episode_data = load_episodes(args.episode_path)
        else:
            # Generate episode path from env parameters
            episode_path = get_episode_file_path(
                env_name=args.env_name,
                benchmark_id=args.benchmark_id,
                checkpoint_step=args.checkpoint,
                episode_num=args.episode_num,
            )
            episode_data = load_episodes(episode_path)

        print(episode_data)
        original_obs = episode_data["obs"]
        original_actions = episode_data["actions"]

        # First, get predictions from inverse projection
        grid_predictions = predictor.predict_for_inverse_projections(
            inverse_projection_file=args.projection_data,
            original_obs=original_obs,
            original_actions=original_actions,
            action_space_size=args.action_space_size,
        )

        # Save combined predictions
        predictor.save_predictions(
            predictions=grid_predictions,
            output_dir=args.output_dir,
            filename_prefix="combined_predictions",
            source_file=args.projection_data,
        )

    else:
        # Process each source individually (original behavior)
        if args.projection_data:
            # Predict for inverse projections
            predictions = predictor.predict_for_inverse_projections(
                inverse_projection_file=args.projection_data, action_space_size=args.action_space_size
            )

            # Save predictions, using the original filename pattern
            predictor.save_predictions(
                predictions=predictions,
                output_dir=args.output_dir,
                filename_prefix="inverse_projection_predictions",
                source_file=args.projection_data,
            )

        if args.episode_path:
            # Load episode data
            episode_data = load_episodes(args.episode_path)

            # Predict for observations in the episode
            predictions = predictor.predict_for_states(
                observations=episode_data["obs"],
                actions=episode_data["actions"] if "actions" in episode_data else None,
                action_space_size=args.action_space_size,
            )

            # Save predictions
            predictor.save_predictions(
                predictions=predictions, output_dir=args.output_dir, filename_prefix="episode_predictions"
            )

        elif (
            args.env_name
            and args.benchmark_id is not None
            and args.checkpoint is not None
            and args.episode_num is not None
        ):
            # Generate episode path
            episode_path = get_episode_file_path(
                env_name=args.env_name,
                benchmark_id=args.benchmark_id,
                checkpoint_step=args.checkpoint,
                episode_num=args.episode_num,
            )

            # Load episode data
            episode_data = load_episodes(episode_path)

            # Predict for observations in the episode
            predictions = predictor.predict_for_states(
                observations=episode_data["obs"],
                actions=episode_data["actions"] if "actions" in episode_data else None,
                action_space_size=args.action_space_size,
            )

            # Save predictions
            predictor.save_predictions(
                predictions=predictions, output_dir=args.output_dir, filename_prefix=f"episode_{args.episode_num}_predictions"
            )

    print("Predictions completed successfully!")


if __name__ == "__main__":
    main()

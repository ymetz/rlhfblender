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
                # For ensemble models, expand inputs to ensemble_count
                obs_tensor = obs_tensor.expand(self.reward_model.ensemble_count, *obs_tensor.shape[1:])
                action_tensor = action_tensor.expand(self.reward_model.ensemble_count, *action_tensor.shape[1:])

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
        self, inverse_projection_file: str, action_space_size: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Load inverse projections and predict rewards and uncertainty

        Args:
            inverse_projection_file: Path to the inverse projection file
            action_space_size: Size of discrete action space (if applicable)

        Returns:
            Dictionary with grid coordinates, predicted rewards, and uncertainty
        """
        # Load inverse projection data
        with open(inverse_projection_file, "r") as f:
            projection_data = json.load(f)

        if "inverse_results" not in projection_data or not projection_data["inverse_results"]:
            raise ValueError("No inverse projection results found in file.")

        grid_samples = projection_data["inverse_results"]["grid_samples"]

        if "reconstructions" not in grid_samples:
            raise ValueError("No reconstructions found in inverse projection results.")

        # Extract reconstructed observations
        reconstructions = np.array(grid_samples["reconstructions"])
        coords = np.array(grid_samples["coords"])
        grid_x = np.array(grid_samples["grid_x"])
        grid_y = np.array(grid_samples["grid_y"])

        # Predict rewards and uncertainty
        if self.policy_model is not None:
            # Predict actions for each reconstructed observation
            actions = self.predict_actions(reconstructions)
        else:
            # Use random actions if no policy model is available
            if action_space_size is None:
                # Assume a default action space size
                action_space_size = 4

            if len(reconstructions.shape) > 3:  # Image observations
                # Probably a discrete action space
                actions = np.random.randint(0, action_space_size, size=reconstructions.shape[0])
            else:
                # Probably a continuous action space with dim=1
                actions = np.random.uniform(-1, 1, size=(reconstructions.shape[0], 1))

        rewards, uncertainty = self.predict_rewards_and_uncertainty(reconstructions, actions, action_space_size)

        return {
            "coords": coords,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "rewards": rewards,
            "uncertainty": uncertainty,
            "actions": actions,
        }

    def save_predictions(
        self, predictions: Dict[str, np.ndarray], output_dir: str, filename_prefix: str = "predictions"
    ) -> str:
        """
        Save predictions to output directory

        Args:
            predictions: Dictionary with prediction results
            output_dir: Directory to save predictions
            filename_prefix: Prefix for output files

        Returns:
            Path to saved predictions file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save as NPZ file
        output_path = os.path.join(output_dir, f"{filename_prefix}.npz")
        np.savez(output_path, **predictions)

        # Also save a JSON summary with basic stats
        summary = {
            "reward_mean": float(np.mean(predictions["rewards"])),
            "reward_std": float(np.std(predictions["rewards"])),
            "reward_min": float(np.min(predictions["rewards"])),
            "reward_max": float(np.max(predictions["rewards"])),
            "uncertainty_mean": float(np.mean(predictions["uncertainty"])) if "uncertainty" in predictions else 0,
            "uncertainty_std": float(np.std(predictions["uncertainty"])) if "uncertainty" in predictions else 0,
            "num_samples": len(predictions["rewards"]),
            "prediction_keys": list(predictions.keys()),
        }

        summary_path = os.path.join(output_dir, f"{filename_prefix}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved predictions to {output_path}")
        print(f"Saved summary to {summary_path}")

        return output_path


def load_episodes(episode_path: str) -> Dict[str, np.ndarray]:
    """
    Load episode data from an NPZ file

    Args:
        episode_path: Path to the episode NPZ file

    Returns:
        Dictionary with episode data
    """
    print(f"Loading episode data from {episode_path}")
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

    # Input data source (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--episode-path", type=str, help="Path to episode NPZ file")
    input_group.add_argument("--inverse-projection", type=str, help="Path to inverse projection JSON file")
    input_group.add_argument(
        "--env-name", type=str, help="Environment name (used with --benchmark-id, --checkpoint-step, --episode-num)"
    )

    # Episode identification (required if --env-name is used)
    parser.add_argument("--benchmark-id", type=int, help="Benchmark ID")
    parser.add_argument("--checkpoint-step", type=int, help="Checkpoint step")
    parser.add_argument("--episode-num", type=int, help="Episode number")

    args = parser.parse_args()

    # Create predictor
    predictor = RewardUncertaintyPredictor(
        reward_model_path=args.reward_model, 
        policy_model_path=args.policy_model, 
        policy_algorithm=args.policy_algorithm, 
        device=args.device
    )

    # Load data and predict based on input source
    if args.inverse_projection:
        # Predict for inverse projections
        predictions = predictor.predict_for_inverse_projections(
            inverse_projection_file=args.inverse_projection, action_space_size=args.action_space_size
        )

        # Save predictions
        predictor.save_predictions(
            predictions=predictions, output_dir=args.output_dir, filename_prefix="inverse_projection_predictions"
        )

    elif args.episode_path:
        # Load episode data
        episode_data = load_episodes(args.episode_path)

        # Predict for observations in the episode
        predictions = predictor.predict_for_states(
            observations=episode_data["obs"],
            actions=episode_data["actions"] if "actions" in episode_data else None,
            action_space_size=args.action_space_size,
        )

        # Save predictions
        predictor.save_predictions(predictions=predictions, output_dir=args.output_dir, filename_prefix="episode_predictions")

    elif args.env_name and args.benchmark_id and args.checkpoint_step and args.episode_num:
        # Generate episode path
        episode_path = get_episode_file_path(
            env_name=args.env_name,
            benchmark_id=args.benchmark_id,
            checkpoint_step=args.checkpoint_step,
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

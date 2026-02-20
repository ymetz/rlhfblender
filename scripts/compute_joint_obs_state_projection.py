#!/usr/bin/env python3
"""
Compute Joint Observation Projections (across checkpoints) and optional Inverse State Projections.

This single-file script merges:
1) Joint projection across multiple checkpoints (e.g., UMAP, PCA, t-SNE)
2) Optional inverse state projection that maps 2D coords -> environment states

It saves:
- Projection handler (pickle)
- Per-checkpoint projected coordinates (npz)
- Projection metadata (json)
- Optional inverse-state model (pkl), combined results (npz), combined metadata (json)
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from databases import Database

# --- RLHFBlender imports (unchanged from your originals) ---
from rlhfblender.data_handling.database_handler import get_single_entry
from rlhfblender.data_models.global_models import Experiment
from rlhfblender.projections.generate_projections import (
    EpisodeID,
    get_available_episodes,
    load_episode_data,
    preprocess_input_data,
    process_env_name,
)
from rlhfblender.projections.projection_handler import ProjectionHandler
from rlhfblender.projections.inverse_state_projection_handler import (
    InverseStateProjectionHandler,
    collect_states_from_multiple_checkpoints,
)

# ------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("joint_obs_state")

# ------------------------------------------------------------------------------------
# Globals & output dirs (kept as in your originals)
# ------------------------------------------------------------------------------------
database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

JOINT_PROJECTIONS_DIR = Path("data/saved_projections/joint")
JOINT_PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

JOINT_OBS_STATE_DIR = Path("data/saved_projections/joint_obs_state")
JOINT_OBS_STATE_DIR.mkdir(parents=True, exist_ok=True)

def compute_margins(all_coords, margin_percent=0.15):
    """
    Compute uniform margins as a percentage of the maximum data range.
    This ensures consistent relative spacing regardless of projection scale.
    
    Args:
        all_coords: numpy array of shape (n_points, 2) with x,y coordinates
        margin_percent: margin as a fraction of the max range (e.g., 0.15 = 15%)
    
    Returns:
        tuple: (global_x_range, global_y_range)
    """
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

    # Use the maximum range to ensure uniform margins
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    # Apply uniform margin based on maximum range
    uniform_margin = max_range * margin_percent
    
    global_x_range = (x_min - uniform_margin, x_max + uniform_margin)
    global_y_range = (y_min - uniform_margin, y_max + uniform_margin)
    
    return global_x_range, global_y_range

# ====================================================================================
# Part 1: Joint projection computer (merged from your second file, now using logging)
# ====================================================================================
class JointProjectionComputer:
    """
    Computes joint projections across multiple checkpoints.
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoints: List[int],
        projection_method: str = "UMAP",
        projection_props: Dict[str, Any] = None,
        sequence_length: int = 1,
        step_range: str = "[]",
        reproject: bool = False,
        append_time: bool = False,
        transition_embedding: bool = True,
        feature_embedding: bool = True,
        additional_gym_packages: List[str] = None,
    ):
        self.experiment_name = experiment_name
        self.checkpoints = checkpoints
        self.projection_method = projection_method
        self.projection_props = projection_props or {}
        self.sequence_length = sequence_length
        self.step_range = step_range
        self.reproject = reproject
        self.append_time = append_time
        self.transition_embedding = transition_embedding
        self.feature_embedding = feature_embedding
        self.additional_gym_packages = additional_gym_packages or []

        # Will be filled during execution
        self.db_experiment = None
        self.env_name = None
        self.environment_config = {}
        self.projection_handler = None

    async def load_experiment_info(self):
        """Load experiment information from database."""
        self.db_experiment = await get_single_entry(
            database,
            Experiment,
            self.experiment_name,
            key_column="exp_name",
        )
        self.env_name = process_env_name(self.db_experiment.env_id)
        self.environment_config = self.db_experiment.environment_config or {}
        logger.info(f"Loaded experiment: {self.experiment_name} (ID: {self.db_experiment.id})")
        logger.info(f"Environment: {self.env_name}")

    async def load_all_episode_data(self) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Load episode data for all checkpoints.

        Returns:
            Tuple of (episode_data_list, checkpoint_episode_counts)
        """
        episode_data_list = []
        checkpoint_episode_counts = []

        for checkpoint in self.checkpoints:
            logger.info(f"Loading data for checkpoint {checkpoint}...")

            # Find available episodes for this checkpoint
            episode_nums = get_available_episodes(experiment=self.db_experiment, checkpoint_step=checkpoint)

            if not episode_nums:
                logger.warning(f"No episodes found for checkpoint {checkpoint}, skipping...")
                episode_data_list.append({})
                checkpoint_episode_counts.append(0)
                continue

            logger.info(f"Found {len(episode_nums)} episodes")

            # Create episode IDs
            episodes = [
                EpisodeID(
                    env_name=self.env_name,
                    benchmark_type="random" if self.db_experiment.framework == "random" else "trained",
                    benchmark_id=self.db_experiment.id,
                    checkpoint_step=checkpoint,
                    episode_num=episode_num,
                )
                for episode_num in episode_nums
            ]

            # Load episode data
            episode_data = await load_episode_data(episodes)
            episode_data_list.append(episode_data)
            checkpoint_episode_counts.append(len(episode_data["obs"]) if "obs" in episode_data else 0)

            logger.info(f"Loaded {checkpoint_episode_counts[-1]} observations for checkpoint {checkpoint}")

        return episode_data_list, checkpoint_episode_counts

    def preprocess_all_data(self, episode_data_list: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess and concatenate data from all checkpoints.

        Returns:
            Tuple of (combined_embedding_input, data_split_indices)
        """
        logger.info("Preprocessing data from all checkpoints...")

        embedding_inputs = []
        data_split_indices = [0]  # start index 0

        # Convert step_range from string to list if provided
        step_range_list = None
        if self.step_range != "[]":
            step_range_list = [int(sr) for sr in self.step_range.strip("[]").split(",")]

        for i, episode_data in enumerate(episode_data_list):
            if not episode_data or "obs" not in episode_data:
                continue

            checkpoint = self.checkpoints[i]
            logger.info(f"Preprocessing checkpoint {checkpoint}...")

            # Preprocess input data for this checkpoint
            embedding_input, feature_input, transition_input, episode_indices = preprocess_input_data(
                episode_data,
                self.sequence_length,
                step_range_list,
                self.append_time,
                self.reproject,
                self.env_name,
                transition_embedding=self.transition_embedding,
                feature_embedding=self.feature_embedding,
            )

            if embedding_input.size > 0:
                embedding_inputs.append(embedding_input)
                data_split_indices.append(data_split_indices[-1] + len(embedding_input))
                logger.info(f"Checkpoint {checkpoint}: {len(embedding_input)} samples")

        if not embedding_inputs:
            raise ValueError("No valid data found across any checkpoints!")

        # Concatenate all embedding inputs
        combined_embedding_input = np.concatenate(embedding_inputs, axis=0)
        logger.info(f"Total combined samples: {len(combined_embedding_input)}")

        # Remove last index (it's the end sentinel)
        return combined_embedding_input, data_split_indices[:-1]

    def fit_joint_projection(self, combined_data: np.ndarray) -> ProjectionHandler:
        """
        Fit the joint projection model on combined data.
        """
        logger.info(f"Fitting joint {self.projection_method} projection...")
        logger.info(f"Data shape: {combined_data.shape}")

        # Initialize projection handler
        handler = ProjectionHandler(projection_method=self.projection_method, projection_props=self.projection_props)

        # Fit the projection on combined data
        projection_suffix = f"joint_{self.experiment_name}_{min(self.checkpoints)}_{max(self.checkpoints)}"
        _ = handler.fit(
            combined_data,
            sequence_length=self.sequence_length,
            step_range=None,  # already applied
            episode_indices=None,
            actions=None,
            suffix=projection_suffix,
        )
        return handler

    def split_projected_data(self, projected_data: np.ndarray, data_split_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Split the projected data back into per-checkpoint arrays.
        """
        checkpoint_projections = {}
        for i, checkpoint in enumerate(self.checkpoints):
            start_idx = data_split_indices[i]
            end_idx = data_split_indices[i + 1] if i + 1 < len(data_split_indices) else len(projected_data)
            if start_idx < end_idx:
                checkpoint_projections[checkpoint] = projected_data[start_idx:end_idx]
                logger.info(f"Checkpoint {checkpoint}: {len(checkpoint_projections[checkpoint])} projected points")
        return checkpoint_projections

    def save_joint_projection(self, handler: ProjectionHandler, checkpoint_projections: Dict[int, np.ndarray]) -> str:
        """
        Save the joint projection model and results. Returns metadata path.
        """
        joint_filename = f"{self.env_name}_{self.db_experiment.id}_joint_{self.projection_method}_{min(self.checkpoints)}_{max(self.checkpoints)}"

        # Save the fitted projection handler
        handler_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_handler.pkl"
        with open(handler_path, "wb") as f:
            pickle.dump(handler, f)
        logger.info(f"Saved projection handler to: {handler_path}")

        # Global bounds across all checkpoints (for plotting consistency)
        all_coords = np.concatenate([proj for proj in checkpoint_projections.values()])
        global_x_range, global_y_range = compute_margins(all_coords, margin_percent=0.15)

        logger.info(f"Computed global bounds: x_range={global_x_range}, y_range={global_y_range}")

        # Save projection results
        results_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_results.npz"
        np.savez(
            results_path,
            checkpoints=np.array(self.checkpoints),
            **{f"checkpoint_{cp}": proj for cp, proj in checkpoint_projections.items()},
        )
        logger.info(f"Saved projection results to: {results_path}")

        # Save metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.db_experiment.id,
            "env_name": self.env_name,
            "checkpoints": self.checkpoints,
            "projection_method": self.projection_method,
            "projection_props": self.projection_props,
            "sequence_length": self.sequence_length,
            "step_range": self.step_range,
            "reproject": self.reproject,
            "append_time": self.append_time,
            "transition_embedding": self.transition_embedding,
            "feature_embedding": self.feature_embedding,
            "handler_path": str(handler_path),
            "results_path": str(results_path),
            "global_x_range": global_x_range,
            "global_y_range": global_y_range,
        }

        metadata_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")

        return str(metadata_path)

    async def compute_joint_projection(self) -> Dict[str, Any]:
        """
        Compute joint projection across checkpoints and return a dict with artifacts.
        """
        try:
            await self.load_experiment_info()
            episode_data_list, checkpoint_episode_counts = await self.load_all_episode_data()

            if all(count == 0 for count in checkpoint_episode_counts):
                raise ValueError("No episode data found for any checkpoint!")

            combined_data, data_split_indices = self.preprocess_all_data(episode_data_list)
            handler = self.fit_joint_projection(combined_data)
            projected_data = handler.get_state()
            checkpoint_projections = self.split_projected_data(projected_data, data_split_indices)
            metadata_path = self.save_joint_projection(handler, checkpoint_projections)

            logger.info("Joint projection computation completed successfully!")
            return {
                "handler": handler,
                "projected_data": projected_data,
                "checkpoint_projections": checkpoint_projections,
                "metadata_path": metadata_path,
                "combined_data_shape": combined_data.shape,
                "data_split_indices": data_split_indices,
                "db_experiment": self.db_experiment,
                "env_name": self.env_name,
                "environment_config": self.environment_config,
            }

        except Exception as e:
            logger.error(f"Error computing joint projection: {e}")
            traceback.print_exc()
            raise


# ====================================================================================
# Part 2: Joint obs + inverse-state orchestrator (from your first file)
# ====================================================================================
class JointObservationStateProjectionComputer:
    """Computes both observation and state projections jointly."""

    def __init__(
        self,
        experiment_name: str,
        checkpoints: List[int],
        projection_method: str = "UMAP",
        projection_props: Dict[str, Any] = None,
        sequence_length: int = 1,
        step_range: str = "[]",
        reproject: bool = False,
        append_time: bool = False,
        transition_embedding: bool = True,
        feature_embedding: bool = True,
        additional_gym_packages: List[str] = None,
        max_trajectories_per_checkpoint: int = 50,
        max_steps_per_trajectory: int = 200,
        state_network_params: Dict[str, Any] = None,
    ):
        self.experiment_name = experiment_name
        self.checkpoints = checkpoints
        self.projection_method = projection_method
        self.projection_props = projection_props or {}
        self.sequence_length = sequence_length
        self.step_range = step_range
        self.reproject = reproject
        self.append_time = append_time
        self.transition_embedding = transition_embedding
        self.feature_embedding = feature_embedding
        self.additional_gym_packages = additional_gym_packages or []
        self.max_trajectories_per_checkpoint = max_trajectories_per_checkpoint
        self.max_steps_per_trajectory = max_steps_per_trajectory

        self.state_network_params = state_network_params or {
            "hidden_dims": [128, 256, 256, 128],
            "learning_rate": 0.001,
            "num_epochs": 100,
            "batch_size": 64,
        }

        # Will be filled during execution
        self.observation_computer: JointProjectionComputer = None
        self.state_projector: InverseStateProjectionHandler = None
        self.env_name = None
        self.environment_config = None

    async def compute_observation_projection(self) -> Dict[str, Any]:
        """Compute joint observation projection across checkpoints."""
        logger.info("Computing joint observation projection...")

        self.observation_computer = JointProjectionComputer(
            experiment_name=self.experiment_name,
            checkpoints=self.checkpoints,
            projection_method=self.projection_method,
            projection_props=self.projection_props,
            sequence_length=self.sequence_length,
            step_range=self.step_range,
            reproject=self.reproject,
            append_time=self.append_time,
            transition_embedding=self.transition_embedding,
            feature_embedding=self.feature_embedding,
            additional_gym_packages=self.additional_gym_packages,
        )

        obs_results = await self.observation_computer.compute_joint_projection()
        self.env_name = obs_results["env_name"]
        self.environment_config = obs_results["environment_config"]

        return obs_results

    def compute_state_projection(self, observation_coords: np.ndarray) -> Dict[str, Any]:
        """Compute inverse state projection using coordinates from observation projection."""
        logger.info("Computing inverse state projection...")

        logger.info("Loading environment states...")
        all_states, _, checkpoint_indices = collect_states_from_multiple_checkpoints(
            db_experiment=self.experiment_name,
            checkpoints=self.checkpoints,
            environment_name=self.env_name,
            environment_config=self.environment_config,
            max_trajectories_per_checkpoint=self.max_trajectories_per_checkpoint,
            max_steps_per_trajectory=self.max_steps_per_trajectory,
            additional_gym_packages=self.additional_gym_packages,
        )

        if not all_states:
            raise ValueError("No environment states could be loaded!")

        logger.info("Matching states to coordinates...")
        min_length = min(len(all_states), len(observation_coords))
        matched_states = all_states[:min_length]
        matched_coords = observation_coords[:min_length]
        logger.info(f"Matched {min_length} state-coordinate pairs")

        logger.info("Training inverse state projection...")
        self.state_projector = InverseStateProjectionHandler(
            hidden_dims=self.state_network_params["hidden_dims"],
            learning_rate=self.state_network_params["learning_rate"],
            num_epochs=self.state_network_params["num_epochs"],
            batch_size=self.state_network_params["batch_size"],
        )

        state_training_history = self.state_projector.fit(
            coords=matched_coords,
            states=matched_states,
            validation_split=0.2,
        )
        logger.info("Inverse state projection training completed")

        consistency_metrics = self._evaluate_consistency(matched_coords, matched_states)
        return {
            "training_history": state_training_history,
            "consistency_metrics": consistency_metrics,
            "matched_data_size": min_length,
            "total_states_found": len(all_states),
            "state_projector": self.state_projector,
        }

    def _evaluate_consistency(self, coords: np.ndarray, true_states: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate consistency between coordinates and reconstructed states."""
        logger.info("Evaluating coordinate-state consistency...")
        n_eval = min(1000, len(coords))
        eval_indices = np.random.choice(len(coords), n_eval, replace=False)
        eval_coords = coords[eval_indices]
        eval_states = [true_states[i] for i in eval_indices]

        predicted_states = self.state_projector.predict(eval_coords)

        component_errors = {}
        total_mse = 0.0
        n_components = 0

        for true_state, pred_state in zip(eval_states, predicted_states):
            if isinstance(true_state, np.ndarray) and len(true_state) == 1:
                true_state = true_state[0]
            if true_state is None or not isinstance(true_state, dict):
                continue

            for key in true_state.keys():
                if key in pred_state:
                    true_val = np.array(true_state[key]).flatten()
                    pred_val = np.array(pred_state[key]).flatten()
                    if len(true_val) == len(pred_val):
                        mse = np.mean((true_val - pred_val) ** 2)
                        component_errors.setdefault(key, []).append(mse)
                        total_mse += mse
                        n_components += 1

        avg_component_errors = {key: float(np.mean(errors)) for key, errors in component_errors.items()}
        overall_mse = total_mse / n_components if n_components > 0 else float("inf")
        metrics = {
            "overall_mse": float(overall_mse),
            "component_errors": avg_component_errors,
            "evaluation_samples": n_eval,
        }
        logger.info(f"Consistency evaluation - Overall MSE: {overall_mse:.6f}")
        return metrics

    def save_joint_results(self, obs_results: Dict[str, Any], state_results: Dict[str, Any]) -> str:
        """Save combined observation and state projection results."""
        joint_filename = (
            f"{self.env_name}_{self.observation_computer.db_experiment.id}"
            f"_joint_obs_state_{self.projection_method}_{min(self.checkpoints)}_{max(self.checkpoints)}"
        )

        # Save the inverse state projection model
        state_model_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_state_model.pkl"
        self.state_projector.save_model(str(state_model_path))
        logger.info(f"Saved state projection model to: {state_model_path}")

        # Global bounds across all checkpoints for consistent scaling
        all_coords = np.concatenate([proj for proj in obs_results["checkpoint_projections"].values()])
        global_x_range, global_y_range = compute_margins(all_coords, margin_percent=0.15)
        logger.info(f"Computed global bounds: x_range={global_x_range}, y_range={global_y_range}")

        # Save combined results (coords + split idx for convenience)
        results_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_results.npz"
        save_dict = {
            "checkpoints": np.array(self.checkpoints),
            "observation_coords": obs_results["projected_data"],
            "data_split_indices": np.array(obs_results["data_split_indices"]),
        }
        for cp, proj in obs_results["checkpoint_projections"].items():
            save_dict[f"obs_checkpoint_{cp}"] = proj
        np.savez(results_path, **save_dict)
        logger.info(f"Saved combined results to: {results_path}")

        metadata = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.observation_computer.db_experiment.id,
            "env_name": self.env_name,
            "checkpoints": self.checkpoints,
            "projection_method": self.projection_method,
            "projection_props": self.projection_props,
            "sequence_length": self.sequence_length,
            "step_range": self.step_range,
            "reproject": self.reproject,
            "append_time": self.append_time,
            "transition_embedding": self.transition_embedding,
            "feature_embedding": self.feature_embedding,
            "max_trajectories_per_checkpoint": self.max_trajectories_per_checkpoint,
            "max_steps_per_trajectory": self.max_steps_per_trajectory,
            "state_network_params": self.state_network_params,
            # Paths
            "observation_projection_metadata": obs_results["metadata_path"],
            "state_model_path": str(state_model_path),
            "combined_results_path": str(results_path),
            # Results summary
            "observation_data_shape": obs_results["combined_data_shape"],
            "state_consistency_mse": state_results["consistency_metrics"]["overall_mse"],
            "matched_state_coordinate_pairs": state_results["matched_data_size"],
            "total_states_found": state_results["total_states_found"],
            # Global bounds for consistent scaling across checkpoints
            "global_x_range": global_x_range,
            "global_y_range": global_y_range,
        }

        metadata_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved combined metadata to: {metadata_path}")

        return str(metadata_path)

    async def compute_joint_projections(self) -> str:
        """
        Compute observation projection, then inverse-state mapping, and save combined artifacts.
        Returns path to combined metadata JSON.
        """
        try:
            logger.info(
                f"Starting joint observation-state projection for {len(self.checkpoints)} checkpoints: {self.checkpoints}"
            )
            logger.info(f"Projection method: {self.projection_method}")

            obs_results = await self.compute_observation_projection()
            state_results = self.compute_state_projection(obs_results["projected_data"])
            metadata_path = self.save_joint_results(obs_results, state_results)

            logger.info("Joint observation-state projection completed successfully!")
            logger.info("\nSummary:")
            logger.info(f"  Observation data shape: {obs_results['combined_data_shape']}")
            logger.info(f"  State consistency MSE: {state_results['consistency_metrics']['overall_mse']:.6f}")
            logger.info(f"  Matched pairs: {state_results['matched_data_size']}")
            return metadata_path

        except Exception as e:
            logger.error(f"Error computing joint projections: {e}")
            traceback.print_exc()
            raise


# ====================================================================================
# CLI
# ====================================================================================
def _parse_checkpoints(maybe_list: List[str]) -> List[int]:
    """
    Accept either: --checkpoints 1000 2000 3000  OR  --checkpoints "1000,2000,3000"
    """
    if len(maybe_list) == 1 and "," in maybe_list[0]:
        return [int(cp.strip()) for cp in maybe_list[0].split(",") if cp.strip()]
    return [int(x) for x in maybe_list]


def main():
    parser = argparse.ArgumentParser(
        description="Compute joint observation projection across checkpoints, and optionally train inverse state projection."
    )

    # Required
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Space-separated or comma-separated checkpoint steps (e.g., --checkpoints 1000 2000 or '1000,2000')",
    )

    # Projection parameters
    parser.add_argument("--projection-method", type=str, default="UMAP", help="Projection method (UMAP, PCA, t-SNE, etc.)")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for projection")
    parser.add_argument("--step-range", type=str, default="[]", help="Range of steps to include, e.g. '[0,100]'")
    parser.add_argument("--reproject", action="store_true", help="Whether to reproject observations")
    parser.add_argument("--append-time", action="store_true", help="Append time information")
    parser.add_argument("--no-transition", action="store_true", help="Skip transition embedding computation")
    parser.add_argument("--no-feature", action="store_true", help="Skip feature embedding computation")
    parser.add_argument("--additional-gym-packages", type=str, nargs="+", default=[], help="Additional gym packages to import")

    # Projection method specific parameters
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for UMAP (if applicable)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance for UMAP (if applicable)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric for projection methods")

    # Inverse state projection toggle & params
    parser.add_argument(
        "--no-include-state",
        dest="include_state",
        action="store_false",
        default=True,
        help="Skip inverse state projection (default: included)",
    )
    parser.add_argument("--max-trajectories", type=int, default=50, help="Max trajectories per checkpoint for state collection")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per trajectory for state collection")
    parser.add_argument("--state-epochs", type=int, default=200, help="Training epochs for state projection")
    parser.add_argument("--state-batch-size", type=int, default=64, help="Batch size for state projection")
    parser.add_argument("--state-learning-rate", type=float, default=0.001, help="Learning rate for state projection")

    args = parser.parse_args()

    checkpoints = _parse_checkpoints(args.checkpoints)
    if len(checkpoints) < 1:
        logger.error("At least 1 checkpoint is required.")
        sys.exit(1)

    # Build projection properties
    projection_props = {"n_components": 2}
    if args.projection_method == "UMAP":
        projection_props.update({"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "metric": args.metric})
    elif args.projection_method == "t-SNE":
        projection_props.update({"perplexity": 30, "early_exaggeration": 12, "learning_rate": 200})

    # If only observation projection:
    if not args.include_state:
        logger.info(f"Computing joint observation projection for checkpoints: {checkpoints}")
        computer = JointProjectionComputer(
            experiment_name=args.experiment_name,
            checkpoints=checkpoints,
            projection_method=args.projection_method,
            projection_props=projection_props,
            sequence_length=args.sequence_length,
            step_range=args.step_range,
            reproject=args.reproject,
            append_time=args.append_time,
            transition_embedding=not args.no_transition,
            feature_embedding=not args.no_feature,
            additional_gym_packages=args.additional_gym_packages,
        )
        try:
            obs_results = asyncio.run(computer.compute_joint_projection())
            print("\nSuccess! Joint projection saved.")
            print(f"Use the metadata file for loading: {obs_results['metadata_path']}")
        except Exception as e:
            logger.error(f"Failed to compute joint projection: {e}")
            sys.exit(1)
        return

    # Otherwise, do obs + inverse-state
    logger.info(f"Computing joint observation + inverse-state projection for checkpoints: {checkpoints}")
    joint = JointObservationStateProjectionComputer(
        experiment_name=args.experiment_name,
        checkpoints=checkpoints,
        projection_method=args.projection_method,
        projection_props=projection_props,
        sequence_length=args.sequence_length,
        step_range=args.step_range,
        reproject=args.reproject,
        append_time=args.append_time,
        transition_embedding=not args.no_transition,
        feature_embedding=not args.no_feature,
        additional_gym_packages=args.additional_gym_packages,
        max_trajectories_per_checkpoint=args.max_trajectories,
        max_steps_per_trajectory=args.max_steps,
        state_network_params={
            "hidden_dims": [128, 256, 256, 128],
            "learning_rate": args.state_learning_rate,
            "num_epochs": args.state_epochs,
            "batch_size": args.state_batch_size,
        },
    )
    try:
        metadata_path = asyncio.run(joint.compute_joint_projections())
        print("\nSuccess! Joint obs+state projections completed.")
        print(f"Combined metadata file: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to compute joint obs+state projections: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

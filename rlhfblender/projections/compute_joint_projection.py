"""
Compute joint projections across multiple checkpoints for RLHFBlender

This script loads episodes from multiple checkpoints, computes a joint projection
(e.g., PCA, UMAP) that maps all data into a common space, and saves both the
fitted projection model and the projected coordinates for each checkpoint.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from databases import Database

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

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

# Create directory for saving joint projections
JOINT_PROJECTIONS_DIR = Path("data/saved_projections/joint")
JOINT_PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)


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

        # Will be filled during execution
        self.db_experiment = None
        self.env_name = None
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
        print(f"Loaded experiment: {self.experiment_name} (ID: {self.db_experiment.id})")
        print(f"Environment: {self.env_name}")

    async def load_all_episode_data(self) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Load episode data for all checkpoints.

        Returns:
            Tuple of (episode_data_list, checkpoint_episode_counts)
        """
        episode_data_list = []
        checkpoint_episode_counts = []

        for checkpoint in self.checkpoints:
            print(f"\nLoading data for checkpoint {checkpoint}...")

            # Find available episodes for this checkpoint
            episode_nums = get_available_episodes(experiment=self.db_experiment, checkpoint_step=checkpoint)

            if not episode_nums:
                print(f"No episodes found for checkpoint {checkpoint}, skipping...")
                episode_data_list.append({})
                checkpoint_episode_counts.append(0)
                continue

            print(f"Found {len(episode_nums)} episodes: {episode_nums}")

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

            print(f"Loaded {checkpoint_episode_counts[-1]} observations for checkpoint {checkpoint}")

        return episode_data_list, checkpoint_episode_counts

    def preprocess_all_data(self, episode_data_list: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess and concatenate data from all checkpoints.

        Returns:
            Tuple of (combined_embedding_input, data_split_indices)
        """
        print("\nPreprocessing data from all checkpoints...")

        embedding_inputs = []
        data_split_indices = [0]  # Track where each checkpoint's data starts/ends

        # Convert step_range from string to list if provided
        step_range_list = None
        if self.step_range != "[]":
            step_range_list = [int(sr) for sr in self.step_range.strip("[]").split(",")]

        for i, episode_data in enumerate(episode_data_list):
            if not episode_data or "obs" not in episode_data:
                continue

            checkpoint = self.checkpoints[i]
            print(f"Preprocessing checkpoint {checkpoint}...")

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
                print(f"Checkpoint {checkpoint}: {len(embedding_input)} samples")

        if not embedding_inputs:
            raise ValueError("No valid data found across any checkpoints!")

        # Concatenate all embedding inputs
        combined_embedding_input = np.concatenate(embedding_inputs, axis=0)
        print(f"\nTotal combined samples: {len(combined_embedding_input)}")

        return combined_embedding_input, data_split_indices[:-1]  # Remove last index

    def fit_joint_projection(self, combined_data: np.ndarray) -> ProjectionHandler:
        """
        Fit the joint projection model on combined data.

        Args:
            combined_data: Combined data from all checkpoints

        Returns:
            Fitted ProjectionHandler
        """
        print(f"\nFitting joint {self.projection_method} projection...")
        print(f"Data shape: {combined_data.shape}")

        # Initialize projection handler
        handler = ProjectionHandler(projection_method=self.projection_method, projection_props=self.projection_props)

        # Fit the projection on combined data
        projection_suffix = f"joint_{self.experiment_name}_{min(self.checkpoints)}_{max(self.checkpoints)}"

        projected_data = handler.fit(
            combined_data,
            sequence_length=self.sequence_length,
            step_range=None,  # Already applied in preprocessing
            episode_indices=None,  # Not using temporal info for joint projection
            actions=None,  # Not using action info for joint projection
            suffix=projection_suffix,
        )

        print(f"Joint projection completed. Output shape: {projected_data.shape}")

        return handler

    def split_projected_data(self, projected_data: np.ndarray, data_split_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        Split the projected data back into per-checkpoint arrays.

        Args:
            projected_data: The joint projected data
            data_split_indices: Indices where each checkpoint's data starts

        Returns:
            Dictionary mapping checkpoint -> projected coordinates
        """
        checkpoint_projections = {}

        for i, checkpoint in enumerate(self.checkpoints):
            start_idx = data_split_indices[i]
            end_idx = data_split_indices[i + 1] if i + 1 < len(data_split_indices) else len(projected_data)

            if start_idx < end_idx:
                checkpoint_projections[checkpoint] = projected_data[start_idx:end_idx]
                print(f"Checkpoint {checkpoint}: {len(checkpoint_projections[checkpoint])} projected points")

        return checkpoint_projections

    def save_joint_projection(self, handler: ProjectionHandler, checkpoint_projections: Dict[int, np.ndarray]) -> str:
        """
        Save the joint projection model and results.

        Args:
            handler: Fitted ProjectionHandler
            checkpoint_projections: Per-checkpoint projected coordinates

        Returns:
            Path to saved joint projection file
        """
        # Create filename for joint projection
        joint_filename = f"{self.env_name}_{self.db_experiment.id}_joint_{self.projection_method}_{min(self.checkpoints)}_{max(self.checkpoints)}"

        # Save the fitted projection handler
        handler_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_handler.pkl"
        with open(handler_path, "wb") as f:
            pickle.dump(handler, f)
        print(f"Saved projection handler to: {handler_path}")

        # Save projection results
        results_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_results.npz"
        np.savez(
            results_path,
            checkpoints=np.array(self.checkpoints),
            **{f"checkpoint_{cp}": proj for cp, proj in checkpoint_projections.items()},
        )
        print(f"Saved projection results to: {results_path}")

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
        }

        metadata_path = JOINT_PROJECTIONS_DIR / f"{joint_filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")

        return str(metadata_path)

    async def compute_joint_projection(self) -> str:
        """
        Main method to compute joint projection across checkpoints.

        Returns:
            Path to saved joint projection metadata file
        """
        try:
            # Load experiment info
            await self.load_experiment_info()

            # Load all episode data
            episode_data_list, checkpoint_episode_counts = await self.load_all_episode_data()

            # Check if we have any data
            if all(count == 0 for count in checkpoint_episode_counts):
                raise ValueError("No episode data found for any checkpoint!")

            # Preprocess and combine data
            combined_data, data_split_indices = self.preprocess_all_data(episode_data_list)

            # Fit joint projection
            handler = self.fit_joint_projection(combined_data)

            # Get projected data
            projected_data = handler.get_state()

            # Split projected data back to per-checkpoint
            checkpoint_projections = self.split_projected_data(projected_data, data_split_indices)

            # Save results
            metadata_path = self.save_joint_projection(handler, checkpoint_projections)

            print(f"\nJoint projection computation completed successfully!")
            print(f"Metadata saved to: {metadata_path}")

            return metadata_path

        except Exception as e:
            print(f"Error computing joint projection: {e}")
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description="Compute joint projections across multiple checkpoints")

    # Required arguments
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--checkpoints", type=str, required=True, help="Comma-separated list of checkpoint steps (e.g., '1000,2000,3000')"
    )

    # Projection parameters
    parser.add_argument("--projection-method", type=str, default="UMAP", help="Projection method (UMAP, PCA, t-SNE, etc.)")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for projection")
    parser.add_argument("--step-range", type=str, default="[]", help="Range of steps to include, e.g. '[0,100]'")
    parser.add_argument("--reproject", action="store_true", help="Whether to reproject observations")
    parser.add_argument("--append-time", action="store_true", help="Append time information")
    parser.add_argument("--no-transition", action="store_true", help="Skip transition embedding computation")
    parser.add_argument("--no-feature", action="store_true", help="Skip feature embedding computation")

    # Projection method specific parameters
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for UMAP (if applicable)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance for UMAP (if applicable)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric for projection methods")

    args = parser.parse_args()

    # Parse checkpoints
    try:
        checkpoints = [int(cp.strip()) for cp in args.checkpoints.split(",")]
    except ValueError:
        print("Error: Invalid checkpoint format. Use comma-separated integers.")
        sys.exit(1)

    if len(checkpoints) < 2:
        print("Error: At least 2 checkpoints are required for joint projection.")
        sys.exit(1)

    print(f"Computing joint projection for {len(checkpoints)} checkpoints: {checkpoints}")

    # Build projection properties
    projection_props = {
        "n_components": 2,  # Always 2D for visualization
    }

    if args.projection_method == "UMAP":
        projection_props.update(
            {
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "metric": args.metric,
            }
        )
    elif args.projection_method == "t-SNE":
        projection_props.update(
            {
                "perplexity": 30,
                "early_exaggeration": 12,
                "learning_rate": 200,
            }
        )

    # Create joint projection computer
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
    )

    # Run computation
    try:
        metadata_path = asyncio.run(computer.compute_joint_projection())
        print(f"\nSuccess! Joint projection saved.")
        print(f"Use the metadata file for loading: {metadata_path}")

    except Exception as e:
        print(f"Failed to compute joint projection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

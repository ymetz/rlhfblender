"""
Generate projections and inverse projections for RLHFBlender

This script loads episodes, computes projections, and also computes inverse projections
with a grid of samples. The results are cached for use in the web interface.
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from databases import Database
from pydantic import BaseModel
from scipy import spatial
from sklearn.cluster import DBSCAN

from rlhfblender.data_handling.database_handler import get_single_entry
from rlhfblender.data_models.global_models import Experiment

# Import the InverseProjectionHandler
from rlhfblender.projections.inverse_projection_handler import InverseProjectionHandler
from rlhfblender.projections.projection_handler import ProjectionHandler

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

# Create directory for saving inverse projection models
INVERSE_MODELS_DIR = Path("data/saved_projections/models")
INVERSE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Create directory for caching inverse projection results
INVERSE_CACHE_DIR = Path("data/saved_projections")
INVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EpisodeID(BaseModel):
    """
    Identifies a specific episode to load data from.
    """

    env_name: str = ""  # e.g.: BreakoutNoFrameskip-v4
    benchmark_type: str = ""  # e.g.: trained
    benchmark_id: int = -1  # e.g.: 1
    checkpoint_step: int = -1  # e.g.: 1000000
    episode_num: int = -1  # e.g.: 0


class InverseProjectionOptions(BaseModel):
    """
    Options for inverse projection computation.
    """

    model_type: str = "auto"  # Options: "auto", "mlp", "cnn", "vae"
    num_epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 64
    validation_split: float = 0.1
    grid_resolution: int = 20
    auto_grid_range: bool = True
    x_range: Tuple[float, float] = (-5.0, 5.0)
    y_range: Tuple[float, float] = (-5.0, 5.0)
    grid_margin: float = 1.5
    force_retrain: bool = False


# Function to process environment name
def process_env_name(env_name: str) -> str:
    """Process environment name to be compatible with filesystem."""
    return env_name.replace("/", "_").replace(":", "_")


# Function to get available episodes for a given configuration
def get_available_episodes(experiment: Experiment, checkpoint_step: int) -> List[int]:
    """
    Find all available episode numbers for a given configuration.

    Args:
        env_name: Environment name
        benchmark_type: Type of benchmark (e.g., 'trained', 'random')
        checkpoint_step: Checkpoint step

    Returns:
        List of available episode numbers
    """

    base_dir = os.path.join(
        "data",
        "episodes",
        process_env_name(experiment.env_id),
        f"{process_env_name(experiment.env_id)}_{experiment.id}_{checkpoint_step}",
    )

    print("Checking for available episodes in:", base_dir)

    # Check if directory exists
    if not os.path.exists(base_dir):
        return []

    # Find all benchmark files
    episode_files = glob.glob(os.path.join(base_dir, "benchmark_*.npz"))

    # Extract episode numbers
    episode_nums = []
    for file_path in episode_files:
        file_name = os.path.basename(file_path)
        try:
            episode_num = int(file_name.split("_")[1].split(".")[0])
            episode_nums.append(episode_num)
        except (IndexError, ValueError):
            continue

    return sorted(episode_nums)


# Function to get episode file path
def get_episode_file_path(episode_id: EpisodeID, data_type: str = "episodes") -> str:
    """
    Generate file path for a specific episode and data type.

    Args:
        episode_id: Episode identifier
        data_type: Type of data to load (episodes, renders, thumbnails, rewards, uncertainty)

    Returns:
        Path to the requested file
    """
    base_dir = os.path.join(
        "data",
        data_type,
        process_env_name(episode_id.env_name),
        f"{process_env_name(episode_id.env_name)}_{episode_id.benchmark_id}_{episode_id.checkpoint_step}",
    )

    if data_type == "episodes":
        return os.path.join(base_dir, f"benchmark_{episode_id.episode_num}.npz")
    elif data_type in ["renders", "thumbnails"]:
        file_ext = "mp4" if data_type == "renders" else "jpg"
        return os.path.join(base_dir, f"{episode_id.episode_num}.{file_ext}")
    elif data_type in ["rewards", "uncertainty"]:
        data_prefix = "rewards" if data_type == "rewards" else "uncertainty"
        return os.path.join(base_dir, f"{data_prefix}_{episode_id.episode_num}.npy")
    else:
        raise ValueError(f"Unknown data type: {data_type}")


# Function to load episode data
async def load_episode_data(episodes: List[EpisodeID]) -> Dict[str, np.ndarray]:
    """
    Load episode data for a list of episode IDs.

    Args:
        episodes: List of episode identifiers

    Returns:
        Dictionary with concatenated episode data
    """
    obs_list = []
    renders_list = []
    actions_list = []
    dones_list = []
    features_list = []
    probs_list = []
    rewards_list = []
    infos_list = []
    episode_steps = []

    for i, episode_id in enumerate(episodes):
        try:
            # Load episode data
            episode_file = get_episode_file_path(episode_id, "episodes")
            print(f"Loading episode data from {episode_file}")
            episode_data = np.load(episode_file, allow_pickle=True)

            # Collect data
            obs_list.append(episode_data["obs"])
            actions_list.append(episode_data["actions"])
            dones_list.append(episode_data["dones"])

            # Add probs if they exist, otherwise use placeholder
            if "probs" in episode_data:
                probs_list.append(episode_data["probs"])
            else:
                # Create placeholder probs based on action space size
                if len(episode_data["actions"].shape) > 1:
                    action_size = episode_data["actions"].shape[1]
                else:
                    action_size = 1
                placeholder_probs = np.zeros((len(episode_data["obs"]), action_size))
                probs_list.append(placeholder_probs)

            rewards_list.append(episode_data["rewards"])
            infos_list.append(episode_data["infos"])

            # Track step indices for temporal information
            steps = np.arange(len(episode_data["obs"]))
            episode_steps.extend(steps.tolist())

            # Try to load features if they exist, otherwise use empty placeholders
            if "features" in episode_data:
                features_list.append(episode_data["features"])
            else:
                # Create empty features with appropriate shape based on obs
                feature_shape = (len(episode_data["obs"]), 128)  # Default feature dimension
                features_list.append(np.zeros(feature_shape))

            # Try to load renders if they exist, otherwise use empty placeholders
            if "renders" in episode_data:
                renders_list.append(episode_data["renders"])
            else:
                # For now, just use empty renders since they're optional
                renders_list.append(np.array([]))

        except Exception as e:
            print(f"Error loading episode {i}: {e}")
            traceback.print_exc()
            continue

    # Concatenate all data (or return empty arrays if no data was loaded)
    result = {
        "obs": np.concatenate(obs_list) if obs_list else np.array([]),
        "actions": np.concatenate(actions_list) if actions_list else np.array([]),
        "dones": np.concatenate(dones_list) if dones_list else np.array([]),
        "features": np.concatenate(features_list) if features_list else np.array([]),
        "probs": np.concatenate(probs_list) if probs_list else np.array([]),
        "rewards": np.concatenate(rewards_list) if rewards_list else np.array([]),
        "infos": np.concatenate(infos_list) if infos_list else np.array([]),
        "episode_steps": np.array(episode_steps) if episode_steps else np.array([]),
        "renders": np.concatenate(renders_list) if renders_list and len(renders_list[0]) > 0 else np.array([]),
    }

    return result


# Function to reproject observations
async def reproject_observations(obs: np.ndarray, env_name: str) -> np.ndarray:
    """
    Reproject observations using a feature extractor.

    Args:
        obs: Observations to reproject
        env_name: Environment name for caching

    Returns:
        Reprojected features
    """
    try:
        # Check if cached reprojected features exist
        reproject_file_name = f"{process_env_name(env_name)}_reproject"

        if os.path.exists(os.path.join("data", "saved_embeddings", reproject_file_name) + ".npy"):
            print(f"Loading cached reprojected features from {reproject_file_name}")
            return np.load(os.path.join("data", "saved_embeddings", reproject_file_name) + ".npy")

        print("Reprojecting observations with feature extractor")

        # Placeholder for actual feature extraction
        # In a real implementation, you would:
        # 1. Import your feature extractor (e.g., ResnetFeatureExtractor)
        # 2. Process observations in batches
        # 3. Return the extracted features

        # For now, simulate the feature extraction with random features
        features = np.random.random((obs.shape[0], 128))

        # Save the features for future use
        os.makedirs(os.path.join("data", "saved_embeddings"), exist_ok=True)
        np.save(os.path.join("data", "saved_embeddings", reproject_file_name), features)

        return features

    except Exception as e:
        print(f"Error in reprojection: {e}")
        traceback.print_exc()
        # Return random features as fallback
        return np.random.random((obs.shape[0], 128))


# Function to preprocess input data for projection
def preprocess_input_data(
    episode_data: Dict[str, np.ndarray],
    sequence_length: int,
    step_range: Optional[List[int]],
    append_time: bool,
    reproject: bool,
    env_name: str,
    transition_embedding: bool = True,
    feature_embedding: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the input data for projection.

    Args:
        episode_data: Dictionary containing episode data
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        append_time: Whether to append temporal information
        reproject: Whether to use reprojected features
        env_name: Environment name (for reprojection caching)

    Returns:
        Tuple containing (embedding_input, feature_input, transition_input, episode_indices)
    """
    obs = episode_data["obs"]
    features = episode_data["features"]
    probs = episode_data["probs"]
    dones = episode_data["dones"]
    renders = episode_data["renders"]

    # Apply step range if specified
    if step_range:
        start, end = step_range
        obs = obs[start:end]
        features = features[start:end]
        probs = probs[start:end]
        if len(dones) > 0:
            dones = dones[start:end]
        if len(renders) > 0:
            renders = renders[start:end]

    # Handle special preprocessing for multi-frame observations (e.g., Atari)
    if len(obs.shape) > 3:
        # ATARI specific, for the first four frames repeat the last frame (others are black by default)
        if len(dones) > 0:
            done_indices = np.argwhere(dones == 1)
            # The first few frames should be initialized
            if obs.shape[0] > 4:
                obs[0:4] = obs[4]
            # After each done state, initialize the next few frames
            for i in done_indices[:-1]:
                obs[i[0] + 1 : i[0] + 4] = obs[i[0] + 4]

        # Subtract the "MEAN" image to keep relevant information (removes static UI elements)
        embedding_input = obs - obs[0]
    else:
        embedding_input = obs

    # Handle reprojection if requested
    if reproject:
        # This would be an async call in real code, but for simplicity we'll make it sync here
        # In real code: embedding_input = await reproject_observations(renders if len(renders) > 0 else obs, env_name)
        loop = asyncio.get_event_loop()
        source = renders if len(renders) > 0 else obs
        embedding_input = loop.run_until_complete(reproject_observations(source, env_name))

    # Prepare episode indices for temporal information
    episode_indices = None
    if append_time:
        episode_indices = episode_data["episode_steps"]
        if len(episode_indices) > 0:
            max_episode_index = np.max(episode_indices)
            if max_episode_index > 0:  # Avoid division by zero
                # Scale the episode indices to match the range of embedding input
                if len(embedding_input.shape) > 1 and embedding_input.shape[0] > 0:
                    max_input_val = np.max(embedding_input[0]) if np.max(embedding_input[0]) > 0 else 1
                    episode_indices = episode_indices * max_input_val / max_episode_index

    # Prepare transition input for transition embedding, if props are 0 everywhere, return None
    if transition_embedding and probs is not None and np.sum(probs) >= 1.0:
        transition_input = probs
    else:
        transition_input = None

    return embedding_input, features, transition_input, episode_indices


# Function to compute projections and clusters
def compute_projections_and_clusters(
    embedding_input: np.ndarray,
    feature_input: np.ndarray,
    transition_input: np.ndarray,
    episode_indices: Optional[np.ndarray],
    sequence_length: int,
    step_range: Optional[List[int]],
    projection_method: str,
    use_one_d_projection: bool,
    actions: np.ndarray,
    projection_props: Dict[str, Any],
    suffix: str,
    joint_projection_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute projections and cluster the projected data.

    Args:
        embedding_input: Input data for main projection
        feature_input: Input features for feature projection
        transition_input: Input transitions for transition projection
        episode_indices: Temporal indices (if append_time is True)
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        projection_method: Projection method to use
        use_one_d_projection: Whether to use 1D projection
        actions: Action data for projection
        projection_props: Additional projection properties
        suffix: Suffix for caching

    Returns:
        Dictionary with projection results
    """
    # Initialize projection handler
    if joint_projection_path:
        print(f"Using joint projection from: {joint_projection_path}")
        handler = ProjectionHandler(
            projection_method=projection_method, projection_props=projection_props, joint_projection_path=joint_projection_path
        )
    else:
        handler = ProjectionHandler(projection_method=projection_method, projection_props=projection_props)

    # Set dimensionality based on option
    if use_one_d_projection:
        handler.embedding_method.n_components = 1
    else:
        handler.embedding_method.n_components = 2

    if joint_projection_path:
        print("Using pre-fitted joint projection model")

    # Compute main projection
    print(f"Computing main projection with method {projection_method}")
    main_projection = handler.fit(
        embedding_input,
        sequence_length=sequence_length,
        step_range=step_range,
        episode_indices=episode_indices,
        actions=actions,
        suffix=suffix,
    )

    print("Computing feature projection")
    if feature_input is not None and feature_input.shape[0] > 0:
        feature_projection = handler.fit(
            feature_input,
            sequence_length=sequence_length,
            step_range=step_range,
            episode_indices=episode_indices,
            actions=None,
            suffix=f"{suffix}_features",
        )
    else:
        feature_projection = np.zeros((0, 2))

    # Compute transition projection using UMAP
    print("Computing transition projection")
    if transition_input is not None and transition_input.shape[0] > 0:
        transition_projection = handler.fit(
            np.squeeze(transition_input),
            sequence_length=sequence_length,
            step_range=step_range,
            episode_indices=None,
            actions=None,
            suffix=f"{suffix}_transition",
        )
    else:
        transition_projection = np.zeros((0, 2))

    # Normalize main projection
    """if main_projection.shape[0] > 0:
        min_val = main_projection.min()
        max_val = main_projection.max()
        if max_val > min_val:
            main_projection = (main_projection - min_val) / (max_val - min_val)

    # Normalize feature projection
    if feature_projection.shape[0] > 0:
        min_val = feature_projection.min()
        max_val = feature_projection.max()
        if max_val > min_val:
            feature_projection = (feature_projection - min_val) / (max_val - min_val)

    # Normalize transition projection
    if transition_projection.shape[0] > 0:
        min_val = transition_projection.min()
        max_val = transition_projection.max()
        if max_val > min_val:
            transition_projection = (transition_projection - min_val) / (max_val - min_val)
    """

    return {
        "projection_array": main_projection,
        "feature_projection": feature_projection,
        "transition_projection": transition_projection,
    }


# Function to cluster and compute point merging
def compute_clusters_and_merging(projection_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute clusters and merged points for the projection.

    Args:
        projection_array: Projected data

    Returns:
        Tuple of (labels, centroids, merged_points, connections)
    """
    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=0.02, min_samples=8)
    dbscan.fit(projection_array)
    labels = dbscan.labels_

    # Compute centroids for each cluster
    unique_labels = np.unique(labels)
    centroids = np.array([np.mean(projection_array[np.where(labels == label)], axis=0) for label in unique_labels])

    # Find connected components for point merging
    components = list(
        nx.connected_components(
            nx.from_edgelist(
                (i, j)
                for i, js in enumerate(spatial.KDTree(projection_array).query_ball_point(projection_array, 0.015))
                for j in js
            )
        )
    )

    # Create clusters mapping
    clusters = {j: i for i, js in enumerate(components) for j in js}

    # Compute merged points with stats
    merged_points = []
    for c in components:
        c_list = list(c)
        if len(c_list) > 0:
            point_data = projection_array[c_list]
            # Compute mean and add shape info
            mean_point = np.mean(point_data, axis=0)
            shape_info = np.array([point_data.shape[0]])  # Number of points in the component
            merged_point = np.concatenate([mean_point, shape_info])
            merged_points.append(merged_point)

    # Convert to numpy array with safe fallback
    if len(merged_points) > 0:
        merged_points = np.array(merged_points)
    else:
        # Create empty array with proper shape if no points
        merged_points = np.zeros((0, projection_array.shape[1] + 1))

    # Compute connections between merged points
    if len(clusters) > 1:
        # Create adjacency matrix
        adj_matrix = np.zeros((len(merged_points), len(merged_points)))
        for i in range(len(clusters) - 1):
            if i in clusters and i + 1 in clusters:
                adj_matrix[clusters[i], clusters[i + 1]] += 1.0

        # Extract non-zero connections
        connections = np.transpose(np.nonzero(adj_matrix))
        connection_values = np.array([adj_matrix[i, j] for i, j in connections])

        # Add connection strengths as third column
        if len(connections) > 0:
            connections = np.concatenate((connections, connection_values[:, None]), axis=1)

        # Compute flow statistics for merged points
        change_in_connections = np.zeros((len(merged_points), 3))
        for i in range(len(merged_points)):
            incoming_connections = np.sum(adj_matrix[:, i])
            outgoing_connections = np.sum(adj_matrix[i, :])
            change_in_connections[i, 0] = incoming_connections
            change_in_connections[i, 1] = outgoing_connections
            change_in_connections[i, 2] = incoming_connections - outgoing_connections

        # Clean extreme values
        change_in_connections[change_in_connections > 5] = 1
        change_in_connections[change_in_connections < -5] = -1

        # Append flow statistics to merged points
        merged_points = np.concatenate((merged_points, change_in_connections), axis=1)
    else:
        # If there are no connections, return empty arrays
        connections = np.zeros((0, 3))

    return labels, centroids, merged_points, connections


# Function to compute projection
async def compute_projection(
    episode_data: Dict[str, np.ndarray],
    projection_method: str = "UMAP",
    sequence_length: int = 1,
    step_range: str = "[]",
    reproject: bool = False,
    use_one_d_projection: bool = False,
    append_time: bool = False,
    projection_props: Dict[str, Any] = {},
    projection_hash: Optional[str] = None,
    env_name: str = "",
    transition_embedding: bool = True,
    feature_embedding: bool = True,
    joint_projection_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute projection for the provided episode data.

    Args:
        episode_data: Dictionary containing loaded episode data
        projection_method: Name of projection method to use
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        reproject: Whether to reproject observations
        use_one_d_projection: Whether to use 1D projection
        append_time: Whether to append time information
        projection_props: Additional projection properties
        projection_hash: Hash for caching projections
        env_name: Environment name for caching

    Returns:
        Dictionary with projection results
    """
    try:
        # Convert step_range from string to list if provided
        step_range_list = None
        if step_range != "[]":
            step_range_list = [int(sr) for sr in step_range.strip("[]").split(",")]

        # Check if cached projection exists
        if projection_hash is not None:
            projection_save_path = os.path.join("data", "saved_projections", f"{projection_hash}.npz")
            if os.path.exists(projection_save_path):
                print(f"Loading cached projection from {projection_save_path}")
                cached_projection = np.load(projection_save_path, allow_pickle=True)
                return {
                    "projection": cached_projection["projection_array"].tolist(),
                    "labels": cached_projection["labels"].tolist(),
                    "centroids": cached_projection["centroids"].tolist(),
                    "merged_points": cached_projection["merged_points"].tolist(),
                    "connections": cached_projection["connections"].tolist(),
                    "feature_projection": cached_projection["feature_projection"].tolist(),
                    "transition_projection": cached_projection["transition_projection"].tolist(),
                    "actions": cached_projection["actions"].tolist(),
                    "dones": cached_projection["dones"].tolist(),
                    "episode_indices": cached_projection["episode_indices"].tolist(),
                }

        # Preprocess input data
        embedding_input, feature_input, transition_input, episode_indices = preprocess_input_data(
            episode_data,
            sequence_length,
            step_range_list,
            append_time,
            reproject,
            env_name,
            transition_embedding=transition_embedding,
            feature_embedding=feature_embedding,
        )

        # Compute projections
        projection_results = compute_projections_and_clusters(
            embedding_input,
            feature_input,
            transition_input,
            episode_indices,
            sequence_length,
            step_range_list,
            projection_method,
            use_one_d_projection,
            episode_data["actions"],
            projection_props,
            projection_hash if projection_hash else "",
            joint_projection_path=joint_projection_path,
        )

        # Extract projection arrays
        projection_array = projection_results["projection_array"]
        feature_projection = projection_results["feature_projection"]
        transition_projection = projection_results["transition_projection"]

        # Compute clusters and merged points
        labels, centroids, merged_points, connections = compute_clusters_and_merging(projection_array)

        episode_indices = []
        current_episode = 0
        for i, done in enumerate(episode_data["dones"]):
            episode_indices.append(current_episode)
            if done:
                current_episode += 1

        # Save projection if hash is provided
        if projection_hash is not None:
            os.makedirs(os.path.join("data", "saved_projections"), exist_ok=True)
            projection_save_path = os.path.join("data", "saved_projections", f"{projection_hash}.npz")
            print(f"Saving projection to {projection_save_path}")

            np.savez(
                projection_save_path,
                projection_array=projection_array,
                labels=labels,
                centroids=centroids,
                merged_points=merged_points,
                connections=connections,
                feature_projection=feature_projection,
                transition_projection=transition_projection,
                actions=episode_data["actions"],
                dones=episode_data["dones"],
                episode_indices=episode_indices,
            )

        # Return results
        return {
            "projection": projection_array.tolist(),
            "labels": labels.tolist(),
            "centroids": centroids.tolist(),
            "merged_points": merged_points.tolist(),
            "connections": connections.tolist(),
            "feature_projection": feature_projection.tolist(),
            "transition_projection": transition_projection.tolist(),
            "actions": episode_data["actions"].tolist(),
            "dones": episode_data["dones"].tolist(),
            "episode_indices": episode_indices,
        }

    except Exception as e:
        print(f"Error computing projection: {e}")
        traceback.print_exc()
        return {
            "projection": [],
            "labels": [],
            "centroids": [],
            "merged_points": [],
            "connections": [],
            "feature_projection": [],
            "transition_projection": [],
            "actions": [],
            "dones": [],
            "episode_indices": [],
        }


# Function to compute inverse projection
def compute_inverse_projection(
    original_data: np.ndarray, coords_2d: np.ndarray, inverse_options: InverseProjectionOptions, cache_key: str
) -> Dict[str, Any]:
    """
    Compute inverse projection and grid of samples.

    Args:
        original_data: Original high-dimensional data
        coords_2d: 2D coordinates from forward projection
        inverse_options: Options for inverse projection
        cache_key: Cache key for storing results

    Returns:
        Dictionary with inverse projection results
    """
    try:
        cache_file = INVERSE_CACHE_DIR / f"{cache_key}.json"
        model_file = INVERSE_MODELS_DIR / f"{cache_key}.pth"

        # Check if we can use cached results
        if not inverse_options.force_retrain and cache_file.exists() and model_file.exists():
            print(f"Loading cached inverse projection from {cache_file}")
            with open(cache_file, "r") as f:
                return json.load(f)

        print("Computing inverse projection")

        # Set up the inverse projection handler
        handler = InverseProjectionHandler(
            model_type=inverse_options.model_type,
            learning_rate=inverse_options.learning_rate,
            batch_size=inverse_options.batch_size,
            num_epochs=inverse_options.num_epochs,
            save_model=True,
            save_dir=str(INVERSE_MODELS_DIR),
            device=None,  # Auto-detect
        )

        # Determine suitable model type based on data shape
        if len(original_data.shape) > 2:  # Image-like data
            model_type = "cnn" if inverse_options.model_type == "auto" else inverse_options.model_type
        else:  # Vector data
            model_type = "mlp" if inverse_options.model_type == "auto" else inverse_options.model_type
            handler.model_type = model_type

        # Train the inverse projection model
        print(f"Training inverse projection model with {inverse_options.num_epochs} epochs")
        history = handler.fit(
            data=original_data, coords=coords_2d, validation_split=inverse_options.validation_split, verbose=True
        )

        # Determine grid ranges
        if inverse_options.auto_grid_range:
            x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
            y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

            # Add margin
            x_margin = (x_max - x_min) * (inverse_options.grid_margin - 1) / 2
            y_margin = (y_max - y_min) * (inverse_options.grid_margin - 1) / 2

            x_range = (x_min - x_margin, x_max + x_margin)
            y_range = (y_min - y_margin, y_max + y_margin)
        else:
            x_range = inverse_options.x_range
            y_range = inverse_options.y_range

        # Generate grid samples
        print(f"Generating grid samples with resolution {inverse_options.grid_resolution}")
        grid_recon, coords, (grid_x, grid_y) = handler.create_latent_space_grid(
            x_range=x_range, y_range=y_range, resolution=inverse_options.grid_resolution, return_coords=True
        )

        # Extract model info
        inverse_model_info = {
            "model_type": model_type,
            "training_history": {
                "train_loss": [float(loss) for loss in history["train_loss"]],
                "val_loss": [float(loss) for loss in history["val_loss"]] if "val_loss" in history else [],
            },
            "data_shape": list(original_data.shape),
            "num_epochs": inverse_options.num_epochs,
            "learning_rate": inverse_options.learning_rate,
            "batch_size": inverse_options.batch_size,
        }

        # Convert grid data to JSON-serializable format
        grid_samples = {
            "reconstructions": grid_recon.tolist(),
            "coords": coords.tolist(),
            "grid_x": grid_x.tolist(),
            "grid_y": grid_y.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "resolution": inverse_options.grid_resolution,
        }

        # Results are also available in the main projection results, just use this
        results = {"inverse_model_info": inverse_model_info, "grid_samples": grid_samples, "model_path": str(model_file)}

        # with open(cache_file, "w") as f:
        #    json.dump(results, f)

        # Save model with the cache key as filename
        torch.save(
            {
                "model_state_dict": handler.model.state_dict(),
                "model_type": model_type,
                "data_shape": original_data.shape,
                "hidden_dims": handler.hidden_dims,
            },
            model_file,
        )

        print(f"Inverse projection model and grid samples saved to {cache_file}")
        return results

    except Exception as e:
        print(f"Error computing inverse projection: {e}")
        traceback.print_exc()
        return {"inverse_model_info": {}, "grid_samples": {}, "model_path": ""}


# Main function
async def generate_projections(
    experiment_id: str,
    checkpoint_step: int,
    projection_method: str = "UMAP",
    sequence_length: int = 1,
    step_range: str = "[]",
    reproject: bool = False,
    use_one_d_projection: bool = False,
    append_time: bool = False,
    projection_props: Dict[str, Any] = {},
    transition_embedding: bool = True,
    feature_embedding: bool = True,
    compute_inverse: bool = False,
    inverse_options: InverseProjectionOptions = InverseProjectionOptions(),
    joint_projection_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate projections for episodes matching the given parameters.

    Args:
        experiment_id: Experiment/Benchmark ID
        checkpoint_step: Checkpoint step
        projection_method: Name of projection method to use
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        reproject: Whether to reproject observations
        use_one_d_projection: Whether to use 1D projection
        append_time: Whether to append time information
        projection_props: Additional projection properties
        transition_embedding: Whether to compute transition embeddings
        feature_embedding: Whether to compute feature embeddings
        compute_inverse: Whether to compute inverse projection
        inverse_options: Options for inverse projection

    Returns:
        Projection results and optionally inverse projection results
    """

    db_experiment: Experiment = await get_single_entry(
        database,
        Experiment,
        experiment_id,
        key_column="exp_name",
    )
    env_name = process_env_name(db_experiment.env_id)

    # Find available episodes
    episode_nums = get_available_episodes(experiment=db_experiment, checkpoint_step=checkpoint_step)

    if not episode_nums:
        print(f"No episodes found for {env_name} (benchmark_id={db_experiment.id}, checkpoint_step={checkpoint_step})")
        return {
            "projection": [],
            "labels": [],
            "centroids": [],
            "merged_points": [],
            "connections": [],
            "feature_projection": [],
            "transition_projection": [],
            "inverse_results": None,
        }

    print(f"Found {len(episode_nums)} episodes: {episode_nums}")

    # Create episode IDs
    episodes = [
        EpisodeID(
            env_name=env_name,
            benchmark_type="random" if db_experiment.framework == "random" else "trained",
            benchmark_id=db_experiment.id,
            checkpoint_step=checkpoint_step,
            episode_num=episode_num,
        )
        for episode_num in episode_nums
    ]

    # Load episode data
    episode_data = await load_episode_data(episodes)

    # Create projection hash for caching
    projection_hash = f"{process_env_name(env_name)}_{db_experiment.id}_{checkpoint_step}_{projection_method}"

    # Compute projection
    projection_results = await compute_projection(
        episode_data=episode_data,
        projection_method=projection_method,
        sequence_length=sequence_length,
        step_range=step_range,
        reproject=reproject,
        use_one_d_projection=use_one_d_projection,
        append_time=append_time,
        projection_props=projection_props,
        projection_hash=projection_hash,
        env_name=env_name,
        transition_embedding=transition_embedding,
        feature_embedding=feature_embedding,
        joint_projection_path=joint_projection_path,
    )

    # Compute inverse projection if requested
    inverse_results = None
    if compute_inverse and len(projection_results["projection"]) > 0:
        print("Computing inverse projection...")

        # Convert projection coordinates to numpy array
        coords_2d = np.array(projection_results["projection"])

        # Get original data for inverse mapping
        original_data = episode_data["obs"]

        # Load global bounds from joint projection if available
        if joint_projection_path and inverse_options.auto_grid_range:
            try:
                print(f"Loading global bounds from joint projection: {joint_projection_path}")
                with open(joint_projection_path, "r") as f:
                    joint_metadata = json.load(f)
                
                if "global_x_range" in joint_metadata and "global_y_range" in joint_metadata:
                    # Use global bounds from joint projection
                    inverse_options.x_range = tuple(joint_metadata["global_x_range"])
                    inverse_options.y_range = tuple(joint_metadata["global_y_range"])
                    inverse_options.auto_grid_range = False  # Use explicit ranges instead of auto
                    print(f"Using global bounds from joint projection: x_range={inverse_options.x_range}, y_range={inverse_options.y_range}")
                else:
                    print("Warning: Global bounds not found in joint projection metadata, using auto-computed bounds")
            except Exception as e:
                print(f"Warning: Failed to load global bounds from joint projection: {e}")

        # Compute inverse projection
        inverse_results = compute_inverse_projection(
            original_data=original_data,
            coords_2d=coords_2d,
            inverse_options=inverse_options,
            cache_key=projection_hash + "_inverse",
        )

    # Add inverse results to projection results
    projection_results["inverse_results"] = inverse_results
    projection_results["projection_hash"] = projection_hash

    return projection_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate projections and inverse projections for RLHFBlender")

    # Episode identification parameters
    parser.add_argument("--experiment-name", type=str, required=True, help="Benchmark/Experiment ID")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint step")

    # Projection parameters
    parser.add_argument("--projection-method", type=str, default="UMAP", help="Projection method to use")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for projection")
    parser.add_argument("--step-range", type=str, default="[]", help="Range of steps to include, e.g. '[0,100]'")
    parser.add_argument("--reproject", action="store_true", help="Whether to reproject observations")
    parser.add_argument("--one-d-projection", action="store_true", help="Use 1D projection instead of 2D")
    parser.add_argument("--append-time", action="store_true", help="Append time information")
    parser.add_argument("--save-to", type=str, default=None, help="Custom path to save the projection")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for UMAP/t-SNE (if applicable)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance for UMAP (if applicable)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric for projection methods")
    parser.add_argument("--no-clustering", action="store_true", help="Skip clustering and point merging steps")
    parser.add_argument("--no-transition", action="store_true", help="Skip transition embedding computation")
    parser.add_argument("--no-feature", action="store_true", help="Skip feature embedding computation")

    # Joint projection parameter
    parser.add_argument(
        "--joint-projection-path",
        type=str,
        default=None,
        help="Path to joint projection metadata file (.json) or handler file (.pkl)",
    )

    # Inverse projection parameters
    parser.add_argument("--compute-inverse", action="store_true", help="Compute inverse projection")
    parser.add_argument(
        "--inverse-model-type", type=str, default="auto", help="Inverse model type ('auto', 'mlp', 'cnn', 'vae')"
    )
    parser.add_argument("--inverse-epochs", type=int, default=30, help="Number of epochs for inverse model training")
    parser.add_argument("--inverse-lr", type=float, default=0.001, help="Learning rate for inverse model")
    parser.add_argument("--inverse-batch-size", type=int, default=64, help="Batch size for inverse model training")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Resolution of the grid for inverse projection")
    parser.add_argument("--auto-grid-range", action="store_true", help="Automatically determine grid range")
    parser.add_argument(
        "--grid-margin", type=float, default=1.5, help="Margin to add around the data when auto-computing grid range"
    )
    parser.add_argument("--x-range", type=str, default="-5,5", help="Range for x-axis in grid, format: 'min,max'")
    parser.add_argument("--y-range", type=str, default="-5,5", help="Range for y-axis in grid, format: 'min,max'")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of inverse model even if cached")

    args = parser.parse_args()

    # Build projection properties from relevant arguments
    params = {
        "UMAP": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "n_components": 2, "metric": args.metric},
        "PCA": {"n_components": 2},
        "t-SNE": {"perplexity": 30, "early_exaggeration": 12, "learning_rate": 200, "n_components": 2},
        "ParametricAngleUMAP": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "n_components": 2,
            "metric": args.metric,
            "action_angle_weight": 0.5,
        },
    }

    # Parse x and y range for grid
    x_range = tuple(float(x) for x in args.x_range.split(","))
    y_range = tuple(float(y) for y in args.y_range.split(","))

    # Create inverse projection options
    inverse_options = InverseProjectionOptions(
        model_type=args.inverse_model_type,
        num_epochs=args.inverse_epochs,
        learning_rate=args.inverse_lr,
        batch_size=args.inverse_batch_size,
        grid_resolution=args.grid_resolution,
        auto_grid_range=args.auto_grid_range,
        x_range=x_range,
        y_range=y_range,
        grid_margin=args.grid_margin,
        force_retrain=args.force_retrain,
    )

    # Run the projection generation
    try:
        print(f"Generating projections for experiment_name={args.experiment_name}, checkpoint_step={args.checkpoint}")
        print(f"Using method: {args.projection_method}, sequence length: {args.sequence_length}")
        print(f"Additional options: reproject={args.reproject}, 1D={args.one_d_projection}, append_time={args.append_time}")

        if args.compute_inverse:
            print(f"Computing inverse projection with model type: {args.inverse_model_type}")
            print(f"Inverse training parameters: epochs={args.inverse_epochs}, batch_size={args.inverse_batch_size}")
            print(f"Grid parameters: resolution={args.grid_resolution}, auto_range={args.auto_grid_range}")

        # Run the main projection generation function
        projection_results = asyncio.run(
            generate_projections(
                experiment_id=args.experiment_name,
                checkpoint_step=args.checkpoint,
                projection_method=args.projection_method,
                sequence_length=args.sequence_length,
                step_range=args.step_range,
                reproject=args.reproject,
                use_one_d_projection=args.one_d_projection,
                append_time=args.append_time,
                projection_props=params[args.projection_method],
                transition_embedding=not args.no_transition,
                feature_embedding=not args.no_feature,
                compute_inverse=args.compute_inverse,
                inverse_options=inverse_options,
                joint_projection_path=args.joint_projection_path,
            )
        )
        save_path = projection_results.get(
            "projection_hash", f"{args.experiment_name}_{args.checkpoint}_{args.projection_method}"
        )

        # Check if we got valid results
        if projection_results["projection"] and len(projection_results["projection"]) > 0:
            print(f"Projection generated with {len(projection_results['projection'])} points")

            # Check inverse results
            if args.compute_inverse and projection_results["inverse_results"]:
                print("Inverse projection successfully computed")
                inverse_model_info = projection_results["inverse_results"]["inverse_model_info"]
                grid_samples = projection_results["inverse_results"]["grid_samples"]
                model_path = projection_results["inverse_results"]["model_path"]

                print(f"Inverse model type: {inverse_model_info.get('model_type', 'unknown')}")
                print(
                    f"Final training loss: {inverse_model_info.get('training_history', {}).get('train_loss', [])[-1] if inverse_model_info.get('training_history', {}).get('train_loss', []) else 'N/A'}"
                )
                print(f"Grid samples: {grid_samples.get('resolution', 0)}x{grid_samples.get('resolution', 0)} grid")
                print(f"Model saved at: {model_path}")

            # Save as JSON for easier analysis (optional)
            import json

            json_path = os.path.join("data", "saved_projections", f"{save_path}.json")
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in projection_results.items():
                if key == "inverse_results":
                    # Already in JSON-serializable format
                    json_results[key] = value
                elif hasattr(value, "tolist"):  # Convert numpy arrays to lists
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value

            with open(json_path, "w") as f:
                json.dump(json_results, f)
            print(f"Results saved as JSON to {json_path}")

            print("Projection generation completed successfully.")
        else:
            print("Warning: Projection generated empty results. Check input data and parameters.")

    except Exception as e:
        print(f"Error generating projections: {e}")
        traceback.print_exc()
        sys.exit(1)

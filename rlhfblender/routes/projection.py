import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from databases import Database
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from rlhfblender.data_handling.database_handler import get_single_entry

# Import projection computation functions
from rlhfblender.data_models.global_models import Experiment
from rlhfblender.projections.generate_projections import (
    EpisodeID,
    compute_projection,
    get_available_episodes,
    load_episode_data,
    process_env_name,
)

# Import projection handlers
from rlhfblender.projections.inverse_projection_handler import InverseProjectionHandler
from rlhfblender.projections.inverse_state_projection_handler import InverseStateProjectionHandler

# Initialize database
database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

# Create router
router = APIRouter(prefix="/projection")

# Initialize logger
logger = logging.getLogger(__name__)

# Create directory for saving inverse projection models
INVERSE_MODELS_DIR = Path("data/saved_projections/models")
INVERSE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Create directory for caching results
CACHE_DIR = Path("data/saved_projections")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ProjectionRequest(BaseModel):
    """
    Request for projection computation.
    """

    projection_hash: int | None = None
    sequence_length: int = 1
    step_range: str = "[]"
    projection_method: str = "UMAP"
    reproject: bool = False
    use_one_d_projection: bool = False
    append_time: bool = False
    episodes: list[EpisodeID] = []
    projection_props: dict[str, Any] = {}


class BenchmarkRetreiveModel(BaseModel):
    """
    Legacy model for backward compatibility.
    """

    env_id: int = -1
    benchmark_id: int = -1
    benchmark_type: str = ""
    checkpoint_step: int = -1


class InverseProjectionParams(BaseModel):
    """
    Parameters specific to inverse projection.
    """

    inverse_model_type: str = "auto"  # Options: "auto", "mlp", "cnn", "vae"
    num_training_epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 64
    validation_split: float = 0.1
    grid_resolution: int = 20
    auto_grid_range: bool = True
    x_range: Tuple[float, float] = (-5.0, 5.0)
    y_range: Tuple[float, float] = (-5.0, 5.0)
    grid_margin: float = 1.5
    use_cached: bool = True
    force_retrain: bool = False


class InverseProjectionResults(BaseModel):
    """
    Results from inverse projection computation.
    """

    forward_projection: Dict[str, Any]
    inverse_model_info: Dict[str, Any]
    grid_samples: Dict[str, Any]
    model_path: str


@router.post("/generate_projection", response_model=dict[str, Any], tags=["PROJECTION"])
async def generate_projection(
    benchmark_id: int,
    checkpoint_step: int,
    projection_method: str = "UMAP",
    sequence_length: int = 1,
    step_range: str = "[]",
    reproject: bool = False,
    use_one_d_projection: bool = False,
    append_time: bool = False,
    projection_props: Dict[str, Any] = {},
):
    """
    Compute projection for all episodes matching the given parameters.

    Args:
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        projection_method: Name of projection method to use
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        reproject: Whether to reproject observations
        use_one_d_projection: Whether to use 1D projection
        append_time: Whether to append time information
        projection_props: Additional projection properties

    Returns:
        Projection results
    """
    # exp from benchmark_id
    db_experiment = await get_single_entry(database, Experiment, benchmark_id)
    env_name = process_env_name(db_experiment.env_id)

    # check if cached file exists (file name: data/saved_projections/envname_exp_id_checkpoint_step_projection_method.json)
    projection_hash = f"{process_env_name(env_name)}_{db_experiment.id}_{checkpoint_step}_{projection_method}"

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

    # Find available episodes
    episode_nums = get_available_episodes(experiment=db_experiment, checkpoint_step=checkpoint_step)

    if not episode_nums:
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

    # Create episode IDs
    episodes = [
        EpisodeID(
            env_name=env_name,
            benchmark_type="random" if db_experiment.framework == "random" else "trained",
            benchmark_id=benchmark_id,
            checkpoint_step=checkpoint_step,
            episode_num=episode_num,
        )
        for episode_num in episode_nums
    ]

    # Load episode data
    episode_data = await load_episode_data(episodes)

    # Compute projection
    forward_projection = await compute_projection(
        episode_data=episode_data,
        projection_method=projection_method,
        sequence_length=sequence_length,
        step_range=step_range,
        reproject=reproject,
        use_one_d_projection=use_one_d_projection,
        append_time=append_time,
        projection_props=projection_props,
        projection_hash=projection_hash,
    )

    return forward_projection


@router.post("/generate_projection_and_inverse", response_model=InverseProjectionResults, tags=["PROJECTION"])
async def generate_projection_and_inverse(
    benchmark_id: int,
    checkpoint_step: int,
    projection_method: str = "UMAP",
    sequence_length: int = 1,
    step_range: str = "[]",
    reproject: bool = False,
    use_one_d_projection: bool = False,
    append_time: bool = False,
    projection_props: Dict[str, Any] = {},
    projection_hash: Optional[int] = None,
    inverse_params: InverseProjectionParams = InverseProjectionParams(),
    background_tasks: BackgroundTasks = None,
):
    """
    Compute forward projection and then train an inverse projection model to map from
    2D coordinates back to the original space. Also generates a grid of samples using
    the inverse model.

    Args:
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        projection_method: Name of projection method to use
        sequence_length: Sequence length for projection
        step_range: Range of steps to include
        reproject: Whether to reproject observations
        use_one_d_projection: Whether to use 1D projection
        append_time: Whether to append time information
        projection_props: Additional projection properties
        projection_hash: Hash for caching projections
        inverse_params: Parameters specific to inverse projection
        background_tasks: Background tasks for async processing

    Returns:
        Forward projection, inverse model info, and grid samples
    """
    try:
        # First, compute the forward projection
        forward_projection = await generate_projection(
            benchmark_id=benchmark_id,
            checkpoint_step=checkpoint_step,
            projection_method=projection_method,
            sequence_length=sequence_length,
            step_range=step_range,
            reproject=reproject,
            use_one_d_projection=use_one_d_projection,
            append_time=append_time,
            projection_props=projection_props,
            projection_hash=projection_hash,
        )

        if not forward_projection["projection"] or len(forward_projection["projection"]) == 0:
            raise HTTPException(
                status_code=400, detail="Forward projection resulted in empty projection. Cannot compute inverse."
            )

        # Get experiment info
        db_experiment = await get_single_entry(database, Experiment, benchmark_id)
        env_name = process_env_name(db_experiment.env_id)

        # Create episode IDs
        episode_nums = get_available_episodes(experiment=db_experiment, checkpoint_step=checkpoint_step)

        episodes = [
            EpisodeID(
                env_name=env_name,
                benchmark_type="random" if db_experiment.framework == "random" else "trained",
                benchmark_id=benchmark_id,
                checkpoint_step=checkpoint_step,
                episode_num=episode_num,
            )
            for episode_num in episode_nums
        ]

        # Get the original data from the episodes
        episode_data = await load_episode_data(episodes)
        original_data = np.array(episode_data["obs"])

        # Get the 2D coordinates from the forward projection
        coords_2d = np.array(forward_projection["projection"])

        # Generate a cache key based on request parameters and data
        cache_key = generate_cache_key(
            benchmark_id=benchmark_id,
            checkpoint_step=checkpoint_step,
            projection_method=projection_method,
            sequence_length=sequence_length,
            step_range=step_range,
            inverse_model_type=inverse_params.inverse_model_type,
            num_training_epochs=inverse_params.num_training_epochs,
            original_data=original_data,
            coords_2d=coords_2d,
        )

        cache_file = CACHE_DIR / f"{cache_key}.json"
        model_file = INVERSE_MODELS_DIR / f"{cache_key}.pth"

        # Check if we can use cached results
        if inverse_params.use_cached and not inverse_params.force_retrain and cache_file.exists() and model_file.exists():
            # Load cached results
            with open(cache_file, "r") as f:
                cached_results = json.load(f)

            return InverseProjectionResults(
                forward_projection=forward_projection,
                inverse_model_info=cached_results["inverse_model_info"],
                grid_samples=cached_results["grid_samples"],
                model_path=str(model_file),
            )

        # Set up the inverse projection handler
        handler = InverseProjectionHandler(
            model_type=inverse_params.inverse_model_type,
            learning_rate=inverse_params.learning_rate,
            batch_size=inverse_params.batch_size,
            num_epochs=inverse_params.num_training_epochs,
            save_model=True,
            save_dir=str(INVERSE_MODELS_DIR),
            device=None,  # Auto-detect
        )

        # Determine suitable model type based on data shape
        if len(original_data.shape) > 2:  # Image-like data
            model_type = "cnn" if inverse_params.inverse_model_type == "auto" else inverse_params.inverse_model_type
        else:  # Vector data
            model_type = "mlp" if inverse_params.inverse_model_type == "auto" else inverse_params.inverse_model_type

        handler.model_type = model_type

        # Train the inverse projection model
        history = handler.fit(
            data=original_data, coords=coords_2d, validation_split=inverse_params.validation_split, verbose=True
        )

        # Determine grid ranges
        if inverse_params.auto_grid_range:
            x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
            y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

            # Add margin
            x_margin = (x_max - x_min) * (inverse_params.grid_margin - 1) / 2
            y_margin = (y_max - y_min) * (inverse_params.grid_margin - 1) / 2

            x_range = (x_min - x_margin, x_max + x_margin)
            y_range = (y_min - y_margin, y_max + y_margin)
        else:
            x_range = inverse_params.x_range
            y_range = inverse_params.y_range

        # Generate grid samples
        grid_recon, coords, (grid_x, grid_y) = handler.create_latent_space_grid(
            x_range=x_range, y_range=y_range, resolution=inverse_params.grid_resolution, return_coords=True
        )

        # Convert grid data to JSON-serializable format
        grid_samples = {
            "reconstructions": grid_recon.tolist(),
            "coords": coords.tolist(),
            "grid_x": grid_x.tolist(),
            "grid_y": grid_y.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "resolution": inverse_params.grid_resolution,
        }

        # Extract model info
        inverse_model_info = {
            "model_type": model_type,
            "training_history": {
                "train_loss": [float(loss) for loss in history["train_loss"]],
                "val_loss": [float(loss) for loss in history["val_loss"]] if "val_loss" in history else [],
            },
            "data_shape": list(original_data.shape),
            "num_epochs": inverse_params.num_training_epochs,
            "learning_rate": inverse_params.learning_rate,
            "batch_size": inverse_params.batch_size,
        }

        # Cache the results
        cache_results = {"inverse_model_info": inverse_model_info, "grid_samples": grid_samples}

        with open(cache_file, "w") as f:
            json.dump(cache_results, f)

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

        return InverseProjectionResults(
            forward_projection=forward_projection,
            inverse_model_info=inverse_model_info,
            grid_samples=grid_samples,
            model_path=str(model_file),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing inverse projection: {str(e)}")


@router.post("/inverse_project_coordinates", tags=["PROJECTION"])
async def inverse_project_coordinates(model_path: str, coords: List[List[float]]):
    """
    Use a trained inverse projection model to map coordinates back to the original space.

    Args:
        model_path: Path to the saved inverse projection model
        coords: List of 2D coordinates to map back

    Returns:
        Original space reconstructions
    """
    try:
        # Load the model
        handler = InverseProjectionHandler()
        handler.load_model(model_path)

        # Convert coords to numpy array
        coords_np = np.array(coords)

        # Generate reconstructions
        reconstructions = handler.predict(coords_np)

        return {"reconstructions": reconstructions.tolist(), "coords": coords}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inverting coordinates: {str(e)}")


@router.get("/get_projection_methods", response_model=list[str], tags=["PROJECTION"])
async def get_projection_methods():
    """
    Return a list of all available projection methods.
    """
    # In a real implementation, this would use your ProjectionHandler
    # For this example, we'll return a fixed list
    return ["UMAP", "PCA", "t-SNE", "ParametricAngleUMAP"]


@router.get("/get_projection_method_params", response_model=dict[str, Any], tags=["PROJECTION"])
async def get_projection_method_params(projection_method: str = "UMAP"):
    """
    Return the parameters for the given projection method.
    """
    params = {
        "UMAP": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "metric": "euclidean"},
        "PCA": {"n_components": 2},
        "t-SNE": {"perplexity": 30, "early_exaggeration": 12, "learning_rate": 200, "n_components": 2},
        "ParametricAngleUMAP": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "action_angle_weight": 0.5,
        },
    }

    return params.get(projection_method, {})


@router.post("/load_grid_projection_data", response_model=Dict[str, Any], tags=["PROJECTION"])
async def load_grid_projection_data(
    benchmark_id: int,
    checkpoint_step: int,
    projection_method: str = "UMAP",
):
    """
    Load pre-computed reward and uncertainty predictions for grid projections.

    Args:
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        projection_method: Name of projection method used
        sequence_length: Sequence length for projection
        step_range: Range of steps included
        reproject: Whether reprojection was used
        use_one_d_projection: Whether 1D projection was used
        append_time: Whether time was appended

    Returns:
        Dictionary with prediction data for grid and original data points
    """
    try:
        # Get experiment info
        db_experiment = await get_single_entry(database, Experiment, benchmark_id)
        env_name = process_env_name(db_experiment.env_id)

        # Generate projection filename (same as in generate_projection)
        projection_hash = f"{process_env_name(env_name)}_{db_experiment.id}_{checkpoint_step}_{projection_method}"

        # First check for a file with the expected naming convention from our script
        prediction_file = Path("data", "saved_projections", f"{projection_hash}_inverse_predictions.json")

        # Also check in the output directory that might have been specified
        alt_prediction_file = Path("results", f"{projection_hash}_inverse_predictions.json")

        # Check if prediction file exists
        if prediction_file.exists():
            with open(prediction_file, "r") as f:
                prediction_data = json.load(f)
                return prediction_data
        elif alt_prediction_file.exists():
            with open(alt_prediction_file, "r") as f:
                prediction_data = json.load(f)
                return prediction_data
        else:
            # Check for any files with similar names in common directories
            possible_locations = [Path("data", "saved_projections"), Path("results"), Path("predictions"), Path("output")]

            for location in possible_locations:
                if location.exists():
                    # Look for files that contain the projection_hash and end with _inverse_predictions.json
                    matching_files = list(location.glob(f"*{projection_hash}*_inverse_predictions.json"))
                    if matching_files:
                        # Use the first match
                        with open(matching_files[0], "r") as f:
                            prediction_data = json.load(f)
                            return prediction_data

            # If we get here, no file was found
            raise HTTPException(
                status_code=404,
                detail=f"No prediction data found for benchmark_id={benchmark_id}, checkpoint_step={checkpoint_step}. "
                f"Run predict_reward_and_uncertainty.py script first to generate predictions.",
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error loading grid projection data: {str(e)}")


@router.post("/load_grid_projection_image", response_model=Dict[str, Any], tags=["PROJECTION"])
async def load_grid_projection_image(
    benchmark_id: int,
    checkpoint_step: int,
    projection_method: str = "UMAP",
    map_type: str = "prediction",  # Options: "prediction", "uncertainty", "both"
):
    """
    Load pre-computed grid projection image. Check if a cached image exists, otherwise compute it
    """

    # load the grid projection data
    prediction_data = await load_grid_projection_data(
        benchmark_id=benchmark_id,
        checkpoint_step=checkpoint_step,
        projection_method=projection_method,
    )

    # grid_coordianates are lists of lists, convert to numpy array
    prediction_data["grid_coordinates"] = np.array(prediction_data["grid_coordinates"])
    prediction_data["original_coordinates"] = np.array(prediction_data["original_coordinates"])

    # Check if the image is already cached
    data_path = CACHE_DIR / f"{benchmark_id}_{checkpoint_step}_{projection_method}_{map_type}.json"
    if data_path.exists():
        image_data = json.loads(data_path.read_text())
    else:
        if map_type == "both":
            # Load both prediction and uncertainty data
            grid_predictions = prediction_data["grid_predictions"]
            grid_uncertainties = prediction_data["grid_uncertainties"]
            original_predictions = prediction_data["original_predictions"]
            original_uncertainties = prediction_data["original_uncertainties"]
            image_data = InverseProjectionHandler.precompute_bivariate_interpolated_surface(
                grid_coords=prediction_data["grid_coordinates"],
                grid_predictions=grid_predictions,
                grid_uncertainties=grid_uncertainties,
                additional_coords=prediction_data["original_coordinates"],
                additional_predictions=original_predictions,
                additional_uncertainties=original_uncertainties,
                resolution=500,
            )
        else:
            grid_value_name = "grid_predictions" if map_type == "prediction" else "grid_uncertainties"
            original_value_name = "original_predictions" if map_type == "prediction" else "original_uncertainties"
            image_data = InverseProjectionHandler.precompute_interpolated_surface(
                grid_coords=prediction_data["grid_coordinates"],
                grid_values=prediction_data[grid_value_name],
                resolution=500,
                additional_coords=prediction_data["original_coordinates"],
                additional_values=prediction_data[original_value_name],
            )

        # Save the image
        with open(data_path, "w") as f:
            json.dump(image_data, f)

    # Add projection bounds from the data
    image_data["projection_bounds"] = {
        "x_min": image_data["x_range"][0],
        "x_max": image_data["x_range"][1],
        "y_min": image_data["y_range"][0],
        "y_max": image_data["y_range"][1],
        "min_val": image_data["min_value"],
        "max_val": image_data["max_value"],
    }
    # normalized original preds and uncertainties
    image_data["original_predictions"] = (
        (np.array(prediction_data["original_predictions"]) - np.min(prediction_data["original_predictions"]))
        / (np.max(prediction_data["original_predictions"]) - np.min(prediction_data["original_predictions"]))
    ).tolist()
    image_data["original_uncertainties"] = (
        (prediction_data["original_uncertainties"] - np.min(prediction_data["original_uncertainties"]))
        / (np.max(prediction_data["original_uncertainties"]) - np.min(prediction_data["original_uncertainties"]))
    ).tolist()

    return image_data


def generate_cache_key(
    benchmark_id: int,
    checkpoint_step: int,
    projection_method: str,
    sequence_length: int,
    step_range: str,
    inverse_model_type: str,
    num_training_epochs: int,
    original_data: np.ndarray,
    coords_2d: np.ndarray,
):
    """
    Generate a cache key based on request parameters and data.

    Args:
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        projection_method: Projection method name
        sequence_length: Sequence length
        step_range: Step range
        inverse_model_type: Inverse model type
        num_training_epochs: Number of training epochs
        original_data: Original data
        coords_2d: 2D coordinates

    Returns:
        Cache key as string
    """
    # Create a hash of the request parameters and data shape
    key_dict = {
        "benchmark_id": benchmark_id,
        "checkpoint_step": checkpoint_step,
        "projection_method": projection_method,
        "sequence_length": sequence_length,
        "step_range": step_range,
        "inverse_model_type": inverse_model_type,
        "num_training_epochs": num_training_epochs,
        "data_shape": list(original_data.shape),
        "data_hash": hashlib.md5(original_data.tobytes()).hexdigest(),
        "coords_hash": hashlib.md5(coords_2d.tobytes()).hexdigest(),
    }

    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


# New State Projection Endpoints

class StateProjectionParams(BaseModel):
    """Parameters for inverse state projection training."""
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    validation_split: float = 0.2
    hidden_dims: List[int] = [128, 256, 256, 128]


class CoordinateToStateRequest(BaseModel):
    """Request to map coordinates to environment states."""
    coordinates: List[List[float]]  # List of [x, y] coordinates
    model_path: str  # Path to trained inverse state projection model


@router.post("/coordinates_to_states", tags=["STATE_PROJECTION"])
async def coordinates_to_states(request: CoordinateToStateRequest):
    """
    Map 2D coordinates to environment states for demo generation.
    
    Args:
        coordinates: List of [x, y] coordinate pairs
        model_path: Path to trained inverse state projection model
        
    Returns:
        List of environment states compatible with SaveResetWrapper
    """
    try:
        # Load trained model
        handler = InverseStateProjectionHandler()
        handler.load_model(request.model_path)
        
        # Convert coordinates to numpy array
        coords_np = np.array(request.coordinates)
        
        # Predict states
        predicted_states = handler.predict(coords_np)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_states = []
        for state in predicted_states:
            serializable_state = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    serializable_state[key] = value.tolist()
                else:
                    serializable_state[key] = value
            serializable_states.append(serializable_state)
        
        return {
            "states": serializable_states,
            "coordinates": request.coordinates,
            "num_states": len(serializable_states)
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting states: {str(e)}")


@router.get("/available_state_models", tags=["STATE_PROJECTION"])
async def get_available_state_models():
    """
    List available inverse state projection models.
    
    Returns:
        List of available model paths
    """
    models_dir = INVERSE_MODELS_DIR / "states"
    models_dir.mkdir(exist_ok=True)
    
    model_files = list(models_dir.glob("*.pkl"))
    model_paths = [str(f.relative_to(Path.cwd())) for f in model_files]
    
    return {
        "available_models": model_paths,
        "count": len(model_paths)
    }


@router.post("/train_state_projection", tags=["STATE_PROJECTION"])
async def train_state_projection_model(
    background_tasks: BackgroundTasks,
    environment_name: str,
    projection_coordinates_path: str,
    output_model_name: str,
    params: StateProjectionParams = StateProjectionParams(),
    max_trajectories: int = 100,
    max_steps_per_trajectory: int = 200
):
    """
    Train an inverse state projection model in the background.
    
    Args:
        environment_name: Name of the environment (e.g., 'metaworld_reach-v2')
        projection_coordinates_path: Path to projection coordinates file
        output_model_name: Name for the output model
        params: Training parameters
        max_trajectories: Maximum trajectories to process
        max_steps_per_trajectory: Maximum steps per trajectory
        
    Returns:
        Task ID for tracking training progress
    """
    # Create output path
    models_dir = INVERSE_MODELS_DIR / "states"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / f"{output_model_name}.pkl"
    
    # Add background training task
    task_id = f"state_projection_{output_model_name}_{hash(environment_name)}"
    
    background_tasks.add_task(
        _train_state_projection_background,
        environment_name=environment_name,
        projection_coordinates_path=projection_coordinates_path,
        output_path=str(output_path),
        params=params,
        max_trajectories=max_trajectories,
        max_steps_per_trajectory=max_steps_per_trajectory,
        task_id=task_id
    )
    
    return {
        "task_id": task_id,
        "status": "started",
        "output_path": str(output_path),
        "message": "State projection model training started in background"
    }


async def _train_state_projection_background(
    environment_name: str,
    projection_coordinates_path: str, 
    output_path: str,
    params: StateProjectionParams,
    max_trajectories: int,
    max_steps_per_trajectory: int,
    task_id: str
):
    """Background task for training state projection model."""
    try:
        import subprocess
        import sys
        
        # Run the training script
        cmd = [
            sys.executable, "-m", "scripts.train_inverse_state_projection",
            "--environment", environment_name,
            "--projection-file", projection_coordinates_path,
            "--output-model", output_path,
            "--max-trajectories", str(max_trajectories),
            "--max-steps", str(max_steps_per_trajectory),
            "--epochs", str(params.num_epochs),
            "--batch-size", str(params.batch_size),
            "--learning-rate", str(params.learning_rate)
        ]
        
        logger.info(f"Starting training task {task_id}: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            logger.info(f"Task {task_id} completed successfully")
        else:
            logger.error(f"Task {task_id} failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Task {task_id} failed with exception: {e}")

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from databases import Database
from fastapi import APIRouter, HTTPException
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


def _build_projection_hash(env_name: str, experiment_id: int, checkpoint_step: int, projection_method: str) -> str:
    return f"{process_env_name(env_name)}_{experiment_id}_{checkpoint_step}_{projection_method}"


def _load_projection_trajectories(
    env_name: str,
    experiment_id: int,
    checkpoint_step: int,
    projection_method: str,
) -> dict[str, Any] | None:
    """Load cached projection trajectories for a checkpoint if available."""

    projection_hash = _build_projection_hash(env_name, experiment_id, checkpoint_step, projection_method)
    projection_path = CACHE_DIR / f"{projection_hash}.npz"

    if not projection_path.exists():
        logger.warning("Projection cache not found for %s", projection_path)
        return None

    try:
        with np.load(projection_path, allow_pickle=True) as data:
            if "projection_array" not in data or "episode_indices" not in data:
                logger.warning("Projection cache missing required arrays: %s", projection_path)
                return None

            projection_array = data["projection_array"]
            episode_indices = data["episode_indices"]

    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load projection cache %s: %s", projection_path, exc)
        return None

    if projection_array.shape[0] != episode_indices.shape[0]:
        logger.warning(
            "Projection cache shape mismatch for %s: points=%s, indices=%s",
            projection_path,
            projection_array.shape,
            episode_indices.shape,
        )
        return None

    trajectories: dict[int, list[list[float]]] = defaultdict(list)
    total_points = 0

    for point, idx in zip(projection_array, episode_indices):
        if point is None:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):  # pragma: no cover - safeguard against malformed points
            continue
        episode_id = int(idx)
        trajectories[episode_id].append([x, y])
        total_points += 1

    ordered = [
        {"episode": episode, "points": pts} for episode, pts in sorted(trajectories.items(), key=lambda item: item[0]) if pts
    ]

    return {
        "episodes": ordered,
        "episode_count": len(ordered),
        "point_count": total_points,
    }


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
    projection_props: dict[str, Any] = {},
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
    # Try to locate a joint projection metadata to ensure coordinates are in the
    # same global frame as any precomputed grid images
    joint_metadata_path = None
    try:
        import glob

        # Prefer joint obs-state metadata, then fall back to observation-only
        joint_metadata_pattern = f"data/saved_projections/joint_obs_state/{db_experiment.env_id}_*_joint_obs_state_{projection_method}_*_metadata.json"
        metadata_files = glob.glob(joint_metadata_pattern)
        if not metadata_files:
            joint_metadata_pattern = (
                f"data/saved_projections/joint/{db_experiment.env_id}_*_joint_{projection_method}_*_metadata.json"
            )
            metadata_files = glob.glob(joint_metadata_pattern)
        if metadata_files:
            joint_metadata_path = max(metadata_files, key=os.path.getctime)
    except Exception as e:
        logger.warning(f"Failed to search for joint metadata: {e}")

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
        joint_projection_path=joint_metadata_path,
    )

    return forward_projection


@router.get("/get_projection_methods", response_model=list[str], tags=["PROJECTION"])
async def get_projection_methods():
    """
    Return a list of all available projection methods.
    """
    # In a real implementation, this would use your ProjectionHandler
    # For this example, we'll return a fixed list
    return ["UMAP", "PCA", "t-SNE"]


@router.get("/get_projection_method_params", response_model=dict[str, Any], tags=["PROJECTION"])
async def get_projection_method_params(projection_method: str = "UMAP"):
    """
    Return the parameters for the given projection method.
    """
    params = {
        "UMAP": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "metric": "euclidean"},
        "PCA": {"n_components": 2},
        "t-SNE": {"perplexity": 30, "early_exaggeration": 12, "learning_rate": 200, "n_components": 2},
    }

    return params.get(projection_method, {})


@router.post("/load_grid_projection_data", response_model=dict[str, Any], tags=["PROJECTION"])
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

        # Check if prediction file exists
        if prediction_file.exists():
            with open(prediction_file) as f:
                prediction_data = json.load(f)
                return prediction_data

        else:
            # If we get here, no file was found
            raise HTTPException(
                status_code=404,
                detail=f"No prediction data found for benchmark_id={benchmark_id}, checkpoint_step={checkpoint_step}. "
                f"Run predict_reward_and_uncertainty.py script first to generate predictions.",
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error loading grid projection data: {e!s}")


def load_global_bounds(benchmark_id: int | None = None, env_id: str | None = None, projection_method: str = "UMAP"):
    """
    Load global bounds from joint projection metadata if available.
    Benchmark_id takes priority over env_id.

    Returns:
        tuple: (global_x_range, global_y_range) or (None, None) if not found
    """

    if benchmark_id is None and env_id is None:
        return None, None

    try:
        # Look for joint projection metadata files
        if benchmark_id is not None:
            joint_metadata_pattern = (
                f"data/saved_projections/joint_obs_state/*_{benchmark_id}_joint_obs_state_{projection_method}_*_metadata.json"
            )
        else:
            joint_metadata_pattern = (
                f"data/saved_projections/joint_obs_state/{env_id}_*_joint_obs_state_{projection_method}_*_metadata.json"
            )
        import glob

        metadata_files = glob.glob(joint_metadata_pattern)

        print("SEARCH PATTERN:", joint_metadata_pattern)
        print("FOUND FILES:", metadata_files)

        if not metadata_files:
            # Try the simpler joint projection pattern
            if benchmark_id is not None:
                joint_metadata_pattern = (
                    f"data/saved_projections/joint/*_{benchmark_id}_joint_{projection_method}_*_metadata.json"
                )
            else:
                joint_metadata_pattern = f"data/saved_projections/joint/{env_id}_*_joint_{projection_method}_*_metadata.json"
            metadata_files = glob.glob(joint_metadata_pattern)

        if metadata_files:
            # Use the most recent metadata file
            latest_file = max(metadata_files, key=os.path.getctime)

            print("[INFO] Load global bounds from:", latest_file)

            with open(latest_file) as f:
                joint_metadata = json.load(f)

            if "global_x_range" in joint_metadata and "global_y_range" in joint_metadata:
                return tuple(joint_metadata["global_x_range"]), tuple(joint_metadata["global_y_range"])

    except Exception as e:
        logger.warning(f"Failed to load global bounds: {e}")

    return None, None


@router.post("/load_grid_projection_image", response_model=dict[str, Any], tags=["PROJECTION"])
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

    print("LOAD GRID PROJECTION IMAGE", benchmark_id, checkpoint_step, projection_method, map_type)

    # load experiment from benchmark_id
    db_experiment = await get_single_entry(database, Experiment, benchmark_id)

    # Load global bounds for consistent scaling across checkpoints
    global_x_range, global_y_range = load_global_bounds(None, db_experiment.env_id, projection_method)
    print(f"Global bounds: {global_x_range}, {global_y_range}")

    # grid_coordianates are lists of lists, convert to numpy array
    prediction_data["grid_coordinates"] = np.array(prediction_data["grid_coordinates"])
    prediction_data["original_coordinates"] = np.array(prediction_data["original_coordinates"])

    # Expand global bounds to include the extents of the original coordinates
    try:
        orig_coords = prediction_data["original_coordinates"]
        if isinstance(orig_coords, np.ndarray):
            oc = orig_coords
        else:
            oc = np.array(orig_coords)
        if oc.size > 0:
            min_x, max_x = float(np.min(oc[:, 0])), float(np.max(oc[:, 0]))
            min_y, max_y = float(np.min(oc[:, 1])), float(np.max(oc[:, 1]))
            if global_x_range is None or global_y_range is None:
                global_x_range = (min_x, max_x)
                global_y_range = (min_y, max_y)
            else:
                gx0, gx1 = global_x_range
                gy0, gy1 = global_y_range
                global_x_range = (min(gx0, min_x), max(gx1, max_x))
                global_y_range = (min(gy0, min_y), max(gy1, max_y))
    except Exception as e:
        logger.warning(f"Failed to expand bounds with original coordinates: {e}")

    # Check if the image is already cached
    data_path = CACHE_DIR / f"{benchmark_id}_{checkpoint_step}_{projection_method}_{map_type}.json"
    if data_path.exists():
        image_data = json.loads(data_path.read_text())
        # Validate cached bounds against original coordinate extents; if they don't
        # cover the coordinates, recompute the image with expanded bounds.
        try:
            oc = prediction_data["original_coordinates"]
            if not isinstance(oc, np.ndarray):
                oc = np.array(oc)
            # Trigger recompute if render version changed
            version_mismatch = image_data.get("render_version", 1) != 2
            if oc.size > 0 and "x_range" in image_data and "y_range" in image_data:
                x_min_c, x_max_c = float(np.min(oc[:, 0])), float(np.max(oc[:, 0]))
                y_min_c, y_max_c = float(np.min(oc[:, 1])), float(np.max(oc[:, 1]))
                xr = tuple(image_data["x_range"]) if isinstance(image_data["x_range"], list) else image_data["x_range"]
                yr = tuple(image_data["y_range"]) if isinstance(image_data["y_range"], list) else image_data["y_range"]
                needs_recompute = version_mismatch or x_min_c < xr[0] or x_max_c > xr[1] or y_min_c < yr[0] or y_max_c > yr[1]
                if needs_recompute:
                    # Expand global ranges to include original coordinates
                    gx0, gx1 = global_x_range if global_x_range else (xr[0], xr[1])
                    gy0, gy1 = global_y_range if global_y_range else (yr[0], yr[1])
                    global_x_range = (min(gx0, x_min_c), max(gx1, x_max_c))
                    global_y_range = (min(gy0, y_min_c), max(gy1, y_max_c))
                    if map_type == "both":
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
                            global_x_range=global_x_range,
                            global_y_range=global_y_range,
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
                            global_x_range=global_x_range,
                            global_y_range=global_y_range,
                        )
                    image_data["render_version"] = 2
                    # Overwrite cache with updated image
                    with open(data_path, "w") as f:
                        json.dump(image_data, f)
        except Exception as e:
            logger.warning(f"Failed to validate cached image bounds: {e}")
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
                global_x_range=global_x_range,
                global_y_range=global_y_range,
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
                global_x_range=global_x_range,
                global_y_range=global_y_range,
            )

        # Save the image
        image_data["render_version"] = 2
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
    # Also expose the original coordinates used for prediction, so the UI
    # can render trajectories exactly in the same coordinate frame as the
    # background image (global/joint space when available).
    try:
        image_data["original_coordinates"] = prediction_data["original_coordinates"].tolist()
    except Exception:
        # prediction_data may already be a list
        try:
            image_data["original_coordinates"] = list(prediction_data["original_coordinates"])
        except Exception:
            image_data["original_coordinates"] = []
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


class UncertaintyDifferenceRequest(BaseModel):
    """Request payload for uncertainty difference computation."""

    benchmark_id: int
    current_checkpoint_step: int
    previous_checkpoint_step: int
    projection_method: str = "UMAP"


@router.post("/load_uncertainty_difference", response_model=dict[str, Any], tags=["PROJECTION"])
async def load_uncertainty_difference(
    request: UncertaintyDifferenceRequest,
):
    """Compute the uncertainty difference between two checkpoints along with trajectory overlays."""

    benchmark_id = request.benchmark_id
    current_checkpoint_step = request.current_checkpoint_step
    previous_checkpoint_step = request.previous_checkpoint_step
    projection_method = request.projection_method

    cache_filename = (
        f"{benchmark_id}_{previous_checkpoint_step}_{current_checkpoint_step}_{projection_method}_uncertainty_diff.json"
    )
    cache_path = CACHE_DIR / cache_filename

    if cache_path.exists():
        try:
            with cache_path.open("r") as cache_file:
                cached_payload = json.load(cache_file)
            return cached_payload
        except Exception as exc:  # pragma: no cover - best effort cache read
            logger.warning("Failed to load cached uncertainty difference: %s", exc)

    if previous_checkpoint_step is None:
        raise HTTPException(status_code=400, detail="previous_checkpoint_step is required")

    if previous_checkpoint_step == current_checkpoint_step:
        raise HTTPException(status_code=400, detail="Checkpoints must be different to compute a difference")

    # Load experiment metadata and cached prediction data
    db_experiment = await get_single_entry(database, Experiment, benchmark_id)
    env_name = process_env_name(db_experiment.env_id)

    try:
        current_prediction = await load_grid_projection_data(
            benchmark_id=benchmark_id,
            checkpoint_step=current_checkpoint_step,
            projection_method=projection_method,
        )
        previous_prediction = await load_grid_projection_data(
            benchmark_id=benchmark_id,
            checkpoint_step=previous_checkpoint_step,
            projection_method=projection_method,
        )
    except HTTPException as exc:
        raise HTTPException(status_code=exc.status_code, detail=f"Failed to load projection data: {exc.detail}") from exc

    grid_coords_current = np.asarray(current_prediction.get("grid_coordinates", []), dtype=float)
    grid_coords_prev = np.asarray(previous_prediction.get("grid_coordinates", []), dtype=float)

    grid_unc_current = np.asarray(current_prediction.get("grid_uncertainties", []), dtype=float)
    grid_unc_prev = np.asarray(previous_prediction.get("grid_uncertainties", []), dtype=float)

    if grid_unc_current.size == 0 or grid_unc_prev.size == 0:
        raise HTTPException(status_code=404, detail="Missing grid uncertainty data for one of the checkpoints")

    # Align grid coordinates across checkpoints; fall back to shared coordinates if grids differ
    if (
        grid_coords_current.shape == grid_coords_prev.shape
        and grid_coords_current.size == grid_coords_prev.size
        and np.allclose(grid_coords_current, grid_coords_prev)
    ):
        aligned_coords = grid_coords_current
        aligned_current = grid_unc_current
        aligned_previous = grid_unc_prev
    else:
        current_map = {tuple(coord): val for coord, val in zip(grid_coords_current, grid_unc_current)}
        previous_map = {tuple(coord): val for coord, val in zip(grid_coords_prev, grid_unc_prev)}
        shared_keys = sorted(set(current_map.keys()) & set(previous_map.keys()))

        if not shared_keys:
            raise HTTPException(status_code=500, detail="No overlapping grid coordinates between checkpoints")

        aligned_coords = np.array(shared_keys, dtype=float)
        aligned_current = np.array([current_map[key] for key in shared_keys], dtype=float)
        aligned_previous = np.array([previous_map[key] for key in shared_keys], dtype=float)

    grid_difference = aligned_current - aligned_previous
    max_abs_diff = float(np.max(np.abs(grid_difference))) if grid_difference.size else 0.0

    if max_abs_diff == 0.0:
        value_range = (-1e-6, 1e-6)
    else:
        value_range = (-max_abs_diff, max_abs_diff)

    global_x_range, global_y_range = load_global_bounds(None, db_experiment.env_id, projection_method)

    def extend_bounds(existing_range: tuple[float, float] | None, values: np.ndarray) -> tuple[float, float] | None:
        if values.size == 0:
            return existing_range
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if existing_range is None:
            return (min_val, max_val)
        return (min(existing_range[0], min_val), max(existing_range[1], max_val))

    original_current = np.asarray(current_prediction.get("original_coordinates", []), dtype=float)
    original_previous = np.asarray(previous_prediction.get("original_coordinates", []), dtype=float)

    combined_original = None
    if original_current.size and original_previous.size:
        combined_original = np.vstack((original_previous, original_current))
    elif original_current.size:
        combined_original = original_current
    elif original_previous.size:
        combined_original = original_previous

    if combined_original is not None and combined_original.size:
        global_x_range = extend_bounds(global_x_range, combined_original[:, 0])
        global_y_range = extend_bounds(global_y_range, combined_original[:, 1])

    diff_image_data = InverseProjectionHandler.precompute_interpolated_surface(
        grid_coords=aligned_coords,
        grid_values=grid_difference,
        resolution=500,
        global_x_range=global_x_range,
        global_y_range=global_y_range,
        cmap="RdBu_r",
        value_range=value_range,
    )

    projection_bounds = {
        "x_min": diff_image_data["x_range"][0],
        "x_max": diff_image_data["x_range"][1],
        "y_min": diff_image_data["y_range"][0],
        "y_max": diff_image_data["y_range"][1],
    }

    statistics_payload = {
        "grid": {
            "min": float(np.min(grid_difference)) if grid_difference.size else 0.0,
            "max": float(np.max(grid_difference)) if grid_difference.size else 0.0,
            "mean": float(np.mean(grid_difference)) if grid_difference.size else 0.0,
            "median": float(np.median(grid_difference)) if grid_difference.size else 0.0,
            "std": float(np.std(grid_difference)) if grid_difference.size else 0.0,
            "fraction_decrease": float(np.mean(grid_difference < 0)) if grid_difference.size else 0.0,
        },
        "current_mean_uncertainty": float(np.mean(aligned_current)) if aligned_current.size else 0.0,
        "previous_mean_uncertainty": float(np.mean(aligned_previous)) if aligned_previous.size else 0.0,
    }

    trajectories_current = _load_projection_trajectories(
        env_name=db_experiment.env_id,
        experiment_id=db_experiment.id,
        checkpoint_step=current_checkpoint_step,
        projection_method=projection_method,
    )
    trajectories_previous = _load_projection_trajectories(
        env_name=db_experiment.env_id,
        experiment_id=db_experiment.id,
        checkpoint_step=previous_checkpoint_step,
        projection_method=projection_method,
    )

    response_payload = {
        "current_checkpoint": current_checkpoint_step,
        "previous_checkpoint": previous_checkpoint_step,
        "difference_image": diff_image_data["image"],
        "projection_bounds": projection_bounds,
        "difference_range": {
            "min": float(value_range[0]),
            "max": float(value_range[1]),
        },
        "difference_stats": statistics_payload,
        "grid": {
            "coordinates": aligned_coords.tolist(),
            "difference": grid_difference.tolist(),
            "current_uncertainty": aligned_current.tolist(),
            "previous_uncertainty": aligned_previous.tolist(),
        },
        "trajectories": {
            "current": trajectories_current,
            "previous": trajectories_previous,
        },
    }

    try:
        with cache_path.open("w") as cache_file:
            json.dump(response_payload, cache_file)
    except Exception as exc:  # pragma: no cover - cache writes are best effort
        logger.warning("Failed to cache uncertainty difference: %s", exc)

    return response_payload


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


class CoordinateToStateRequest(BaseModel):
    """Request to map coordinates to environment states."""

    coordinates: list[list[float]]  # List of [x, y] coordinates
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

        return {"states": serializable_states, "coordinates": request.coordinates, "num_states": len(serializable_states)}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting states: {e!s}")

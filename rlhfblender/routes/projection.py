import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from databases import Database

# Import projection computation functions
# In a real implementation, these would be imported from your module
# For this example, we'll import from the standalone script
from rlhfblender.projections.generate_projections import (
    get_available_episodes,
    load_episode_data,
    compute_projection,
    process_env_name,
    EpisodeID
)

# Initialize database
database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

# Create router
router = APIRouter(prefix="/projection")

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
    # In a real implementation, this would use your ProjectionHandler
    # For this example, we'll return fixed parameter sets
    params = {
        "UMAP": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean"
        },
        "PCA": {
            "n_components": 2
        },
        "t-SNE": {
            "perplexity": 30,
            "early_exaggeration": 12,
            "learning_rate": 200,
            "n_components": 2
        },
        "ParametricAngleUMAP": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "action_angle_weight": 0.5
        }
    }
    
    return params.get(projection_method, {})


@router.post("/current_obs_to_projection", response_model=dict[str, Any], tags=["PROJECTION"])
async def get_projection(request: ProjectionRequest):
    """
    Get the projection for a list of episodes.
    
    Args:
        request: Projection request parameters including episode IDs
        
    Returns:
        Projection data
    """
    if not request.episodes:
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
    
    try:
        # Load episode data
        episode_data = await load_episode_data(request.episodes)
        
        # Compute projection
        return await compute_projection(
            episode_data=episode_data,
            projection_method=request.projection_method,
            sequence_length=request.sequence_length,
            step_range=request.step_range,
            reproject=request.reproject,
            use_one_d_projection=request.use_one_d_projection,
            append_time=request.append_time,
            projection_props=request.projection_props,
            projection_hash=request.projection_hash
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing projection: {e!s}") from e


@router.get("/get_available_episodes", response_model=list[int], tags=["PROJECTION"])
async def api_get_available_episodes(
    env_name: str,
    benchmark_type: str,
    benchmark_id: int,
    checkpoint_step: int
):
    """
    Find all available episode numbers for a given configuration.
    
    Args:
        env_name: Environment name
        benchmark_type: Type of benchmark (e.g., 'trained', 'random')
        benchmark_id: Benchmark ID
        checkpoint_step: Checkpoint step
        
    Returns:
        List of available episode numbers
    """

    env_name = process_env_name(env_name)

    return get_available_episodes(
        env_name=env_name,
        benchmark_type=benchmark_type,
        benchmark_id=benchmark_id,
        checkpoint_step=checkpoint_step
    )


@router.post("/generate_projection", response_model=dict[str, Any], tags=["PROJECTION"])
async def generate_projection(
    env_name: str,
    benchmark_type: str,
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
):
    """
    Compute projection for all episodes matching the given parameters.
    
    Args:
        env_name: Environment name
        benchmark_type: Type of benchmark (e.g., 'trained', 'random')
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
        
    Returns:
        Projection results
    """

    env_name = process_env_name(env_name)

    # Find available episodes
    episode_nums = get_available_episodes(
        env_name=env_name,
        benchmark_type=benchmark_type,
        benchmark_id=benchmark_id,
        checkpoint_step=checkpoint_step
    )
    
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
            benchmark_type=benchmark_type,
            benchmark_id=benchmark_id,
            checkpoint_step=checkpoint_step,
            episode_num=episode_num
        )
        for episode_num in episode_nums
    ]
    
    # Create request
    request = ProjectionRequest(
        projection_hash=projection_hash,
        sequence_length=sequence_length,
        step_range=step_range,
        projection_method=projection_method,
        reproject=reproject,
        use_one_d_projection=use_one_d_projection,
        append_time=append_time,
        episodes=episodes,
        projection_props=projection_props
    )
    
    # Forward to main endpoint
    return await get_projection(request)
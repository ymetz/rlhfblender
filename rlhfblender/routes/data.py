import base64
import os
from enum import Enum
from typing import Any

import cv2
import numpy as np
from databases import Database
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rlhfblender.data_collection import framework_selector
from rlhfblender.data_collection.episode_recorder import (
    BenchmarkSummary,
)
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.feedback_models import UnprocessedFeedback
from rlhfblender.data_models.global_models import (
    Environment,
    EpisodeID,
    Experiment,
)
from rlhfblender.utils import convert_to_serializable, process_env_name

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

router = APIRouter(prefix="/data")


@router.get("/get_available_frameworks", response_model=list[str])
def get_available_frameworks():
    """
    Return a list of all available frameworks
    :return:
    """
    return framework_selector.get_framework_list()


@router.get("/get_algorithms", response_model=list[str])
def get_algorithms(selected_framework: str):
    """
    Return a list of all available algorithms for the given framework
    :param selected_framework:
    :return:
    """
    return framework_selector.get_connector(selected_framework).get_algorithms()


class BenchmarkModel(BaseModel):
    """
    A request model for a single benchmark run
    """

    env_id: str = ""
    benchmark_id: str = ""
    checkpoint_step: int = -1


class VideoRequestModel(BenchmarkModel):
    episode_id: int = -1


class BenchmarkResponseModel(BaseModel):
    n_steps: int
    models: list[BenchmarkSummary]
    env_space_info: dict


# Feedback Type Enum
class FeedbackType(str, Enum):
    rating = "rating"
    ranking = "ranking"
    demonstration = "demonstration"
    correction = "correction"
    description = "description"
    text = "text"


class FeedbackModel(BenchmarkModel):
    episode_id: int = -1
    feedback: dict | None = None
    feedback_type: FeedbackType = FeedbackType.rating
    feedback_time: float = -1.0


@router.get("/get_rewards", response_model=list, tags=["DATA"])
async def get_rewards(
    env_name: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """Return step rewards a list for the selected episode"""
    # Replace with your rewards file path
    rewards = np.load(
        os.path.join(
            "data",
            "rewards",
            process_env_name(env_name),
            f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}",
            f"rewards_{episode_num}.npy",
        ),
    )

    return rewards.tolist()


@router.get("/get_uncertainty", response_model=list, tags=["DATA"])
async def get_uncertainty(
    env_name: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """Return step rewards a list for the selected episode"""
    # Replace with your rewards file path
    uncertainty = np.load(
        os.path.join(
            "data",
            "uncertainty",
            process_env_name(env_name),
            f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}",
            f"uncertainty_{episode_num}.npy",
        ),
    )

    return uncertainty.tolist()


@router.get("/get_video", response_class=FileResponse)
async def get_video(
    env_name: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    # Replace with your video file path
    return FileResponse(
        os.path.join(
            "data",
            "renders",
            process_env_name(env_name),
            f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}",
            f"{episode_num}.mp4",
        ),
        media_type="video/mp4",
    )


@router.get("/get_thumbnail", response_class=FileResponse)
async def get_thumbnail(
    env_name: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """
    Should return a telling thumbnail at some point, right now just fetches
    the first frame of an episode.
    """
    return FileResponse(
        os.path.join(
            "data",
            "thumbnails",
            process_env_name(env_name),
            f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}",
            f"{episode_num}.jpg",
        ),
        media_type="image/jpg",
    )


class DetailRequest(BaseModel):
    env_name: str
    benchmark_id: int
    checkpoint_step: int
    episode_num: int


class SingleStepDetailRequest(DetailRequest):
    step: int


@router.post("/get_single_step_details", response_model=dict[str, Any], tags=["DATA"])
async def get_single_step_details(request: SingleStepDetailRequest):
    """
    Returns a dictionary containing all the data for a single step: the action distribution, the action, the reward,
    the info dict, and action space.
    """
    action_space = {}
    db_env = await db_handler.get_single_entry(database, Environment, key=request.env_name, key_column="env_name")
    if db_env is not None:
        action_space = db_env.action_space_info

    # Get the action distribution
    episode_benchmark_data = np.load(
        os.path.join(
            "data",
            "episodes",
            process_env_name(request.env_name),
            f"{process_env_name(request.env_name)}_{request.benchmark_id}_{request.checkpoint_step}",
            f"benchmark_{request.episode_num}.npz",
        ),
        allow_pickle=True,
    )

    try:
        action_distribution = episode_benchmark_data["probs"][request.step]
    except IndexError:
        action_distribution = [0.0]
    action = episode_benchmark_data["actions"][request.step]
    reward = episode_benchmark_data["rewards"][request.step]
    info = episode_benchmark_data["infos"][request.step]

    return convert_to_serializable(
        {
            "action_distribution": action_distribution,
            "action": action,
            "reward": reward,
            "info": info,
            "action_space": action_space,
        }
    )


@router.post("/get_actions_for_episode", response_model=list[int | float | list[float]], tags=["DATA"])
async def get_actions_for_episode(request: DetailRequest):
    """
    Returns a list of all actions for a given episode
    """
    episode_benchmark_data = np.load(
        os.path.join(
            "data",
            "episodes",
            process_env_name(request.env_name),
            f"{process_env_name(request.env_name)}_{request.benchmark_id}_{request.checkpoint_step}",
            f"benchmark_{request.episode_num}.npz",
        ),
        allow_pickle=True,
    )

    return episode_benchmark_data["actions"].tolist()


class SaveFeatureFeedbackRequest(BaseModel):
    session_id: str


@router.post("/save_feature_feedback")
async def save_feature_feedback(request: Request, image: UploadFile = None):

    save_image_name = request.query_params.get("save_image_name", None)

    image = image or File(...)
    import base64
    import io

    from PIL import Image

    contents = await image.read()

    contents_str = contents.decode("utf-8")

    # Remove the first part of the string "data:image/png;base64,"
    contents_str = contents_str.replace("data:image/png;base64,", "")

    img = Image.open(io.BytesIO(base64.b64decode(contents_str)))

    # Save image
    os.makedirs(os.path.join("logs", "feature_feedback"), exist_ok=True)
    img.save(os.path.join("logs", "feature_feedback", save_image_name + ".png"))

    return {"message": "Image saved successfully"}


class ActionLabelRequest(BaseModel):
    envId: str


class ClusterFrameRequest(BaseModel):
    """Request model for extracting frames from cluster selection"""

    cluster_indices: list[int]  # List of step indices within their respective episodes
    episode_data: list[dict]  # List of episode information for each trajectory
    max_states_to_show: int = 12


@router.post("/get_action_label_urls", response_model=list[str])
async def get_action_label_urls(request: ActionLabelRequest):
    """
    Returns a list of urls for the action labels of the given environment
    """
    db_env = await db_handler.get_single_entry(database, Environment, key=request.envId, key_column="registration_id")
    if db_env is None:
        return []
    db_env_name = process_env_name(db_env.env_name)

    # Check in data/action_labels/<env_name> for all files
    action_label_dir = os.path.join("data", "action_labels", db_env_name)
    if not os.path.isdir(action_label_dir):
        return []
    action_label_files = [file for file in os.listdir(action_label_dir) if file.endswith(".png") or file.endswith(".svg")]

    # Return the urls
    return [f"/action_labels/{db_env_name}/{file}" for file in action_label_files]


@router.post("/get_cluster_frames", response_model=list[str])
async def get_cluster_frames(request: ClusterFrameRequest):
    """
    Extract frames from videos for cluster selection.
    Returns base64-encoded images for the specified steps from each trajectory.
    Each cluster_index corresponds to a step within its respective episode.
    """
    try:
        frame_images = []

        # Process each cluster index with its corresponding episode data
        for cluster_index, episode_info in zip(request.cluster_indices, request.episode_data):
            # Extract episode details
            env_name = episode_info.get("env_name", "")
            benchmark_id = episode_info.get("benchmark_id", 0)
            checkpoint_step = episode_info.get("checkpoint_step", 0)
            episode_num = episode_info.get("episode_num", 0)

            # Construct video path using same pattern as get_video endpoint
            video_path = os.path.join(
                "data",
                "renders",
                process_env_name(env_name),
                f"{process_env_name(env_name)}_{benchmark_id}_{checkpoint_step}",
                f"{episode_num}.mp4",
            )

            if not os.path.exists(video_path):
                # Skip if video doesn't exist but log the issue
                print(f"Warning: Video not found at {video_path}")
                continue

            # Extract frame at the specified step index within this episode
            frame_base64 = extract_frame_from_video(video_path, cluster_index)
            if frame_base64:
                frame_images.append(frame_base64)

        return frame_images

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting frames: {str(e)}")


def extract_frame_from_video(video_path: str, frame_index: int) -> str | None:
    """
    Extract a specific frame from a video file and return as base64-encoded string.
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR to RGB
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        return f"data:image/jpeg;base64,{frame_base64}"

    except Exception as e:
        print(f"Error extracting frame from {video_path} at index {frame_index}: {e}")
        return None


@router.post("/reset_sampler")
async def reset_sampler(request: Request):
    """
    Resets the sampler for the given experiment
    """
    experiment_id = request.query_params.get("experiment_id", None)
    sampling_strategy = request.query_params.get("sampling_strategy", None)

    if experiment_id is None:
        return "No experiment id given"

    experiment: Experiment = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    environment = await db_handler.get_single_entry(database, Environment, key=experiment.env_id, key_column="registration_id")

    session_id = await request.app.state.logger.reset(experiment, environment)

    print("Resetting sampler:", experiment_id, experiment.env_id, sampling_strategy)

    request.app.state.sampler.set_sampler(
        experiment, environment, request.app.state.logger, sampling_strategy=sampling_strategy
    )

    request.app.state.feedback_translator.set_translator(experiment, environment, request.app.state.logger)

    return {
        "session_id": session_id,
        "environment_id": experiment.env_id,
    }


@router.post("/step_sampler")
async def step_sampler(request: Request):
    """
    Steps the sampler to return the next batch of episodes
    """
    session_id = request.query_params.get("session_id", None)
    exp_id = request.query_params.get("experiment_id", None)

    if session_id is None or exp_id is None:
        return "No session id or experiment id given"

    # if trained, the database exp. will have an updated checkpoint list
    updated_experiment = await db_handler.get_single_entry(database, Experiment, key=exp_id)

    if updated_experiment is not None:
        request.app.state.sampler.step_sampler(updated_experiment)


@router.get("/get_all_episodes", response_model=list[EpisodeID])
async def get_all_episodes(request: Request):
    """
    Returns all episodes for the current configuration of the sampler
    :param request:
    :return:
    """
    return request.app.state.sampler.get_full_episode_list()


@router.post("/give_feedback")
async def give_feedback(request: Request):
    """
    Provides feedback for a given episode
    """
    ui_feedback = await request.json()
    if not ui_feedback:
        return "Empty feedback"

    for feedback in ui_feedback:
        feedback = UnprocessedFeedback(**feedback)
        request.app.state.feedback_translator.give_feedback(feedback.session_id, feedback)

    return "Feedback received"


# Store for active human-in-the-loop training sessions (shared with dynamic_rlhf module)
active_rlhf_sessions = {}


@router.post("/submit_session")
async def submit_current_feedback(request: Request):
    """
    Submits the current feedback and call post-processing (duplication, common format, etc)
    Saves processed feedback for later integration with DynamicRLHF training.
    """
    session_id = request.query_params.get("session_id", None)
    save_dynamic_rlhf = request.query_params.get("saveDynamicRLHFFormat", "false").lower() == "true"

    if session_id is None:
        return "No session id given"

    # Process feedback through the feedback translator
    processed_feedback = request.app.state.feedback_translator.process()

    print(f"Current session_id: {session_id}")
    print(f"Processed feedback count: {len(processed_feedback) if processed_feedback else 0}")

    if processed_feedback is not None and save_dynamic_rlhf:
        # Save processed feedback as pickle file for training integration
        feedback_dir = f"sessions/{session_id}"
        os.makedirs(feedback_dir, exist_ok=True)

        # Also save in DynamicRLHF format for direct consumption by training
        from rlhfblender.data_collection.feedback_dataset_adapter import FeedbackDatasetAdapter

        dynamic_rlhf_file = os.path.join(feedback_dir, "dynamic_rlhf_feedback.pkl")
        success = FeedbackDatasetAdapter.save_dynamic_rlhf_format(processed_feedback, dynamic_rlhf_file)
        if success:
            print(f"Successfully saved DynamicRLHF format feedback to {dynamic_rlhf_file}")
        else:
            print("Failed to save DynamicRLHF format feedback")
    else:
        print("No processed feedback to save")

    return "Feedback submitted"

import asyncio
import base64
import os
import random
import tempfile
import time
import uuid
from enum import Enum
from typing import Any

import cv2
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.rtcconfiguration import RTCIceServer
from databases import Database
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rlhfblender.data_collection import framework_selector, webrtc_demo_session
from rlhfblender.data_collection.demo_session import (
    create_new_session,
    demo_perform_step,
)
from rlhfblender.data_collection.episode_recorder import (
    BenchmarkSummary,
)
from rlhfblender.data_collection.feedback_translator import FeedbackTranslator
from rlhfblender.data_collection.reward_learning_handler import RewardModelHandler
from rlhfblender.data_collection.webrtc_demo_session import GymEnvironmentTrack
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

# Simulation store for web demo
simulation_store = {}


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

    action_distribution = episode_benchmark_data["probs"][request.step]
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


@router.post("/submit_session")
async def submit_current_feedback(request: Request):
    """
    Submits the current feedback and call post-processing (duplication, common format, etc)
    """
    session_id = request.query_params.get("session_id", None)
    if session_id is None:
        return "No session id given"
    request.app.state.feedback_translator.process(session_id)
    return "Feedback submitted"


@router.post("/initialize_demo_session")
async def initialize_demo_session(request: Request):
    """
    To generate demos, we initialize a gym environment in a separate process, then communicate with it via
    a socket. This function initializes the environment via a gym id, optional seed and returns the port
    :param request:
    :return:
    """
    request = await request.json()
    env_id = request.get("env_id", None)
    exp_id = request.get("exp_id", None)
    print("EXP ID", exp_id, env_id)
    seed = request["seed"]
    session_id = request["session_id"]

    action_space = {}
    exp = await db_handler.get_single_entry(database, Experiment, key=exp_id)
    db_env = await db_handler.get_single_entry(database, Environment, key=env_id, key_column="registration_id")
    if db_env is not None:
        action_space = db_env.action_space_info

    try:
        pid, demo_number = await create_new_session(session_id, exp, db_env, int(seed))

        first_step = demo_perform_step(session_id, [])
        success = True
    except Exception:
        pid = -1
        first_step = {"reward": 0, "done": False, "infos": {}}
        success = False

    return {
        "pid": pid,
        "demo_number": demo_number,
        "action_space": action_space,
        "step": first_step,
        "success": success,
    }


@router.post("/demo_step")
async def demo_step(request: Request):
    """
    Performs a step in the demo environment
    :param request:
    :return:
    """
    request = await request.json()
    session_id = request["session_id"]
    action = request["action"]

    try:
        return_data = demo_perform_step(session_id, action)
        success = True
    except Exception:
        return_data = {"reward": 0, "done": False, "infos": {}}
        success = False

    return {"step": return_data, "success": success}


@router.post("/end_demo_session")
async def end_demo_session(request: Request):
    """
    Closes the demo session
    :param request:
    :return:
    """
    request = await request.json()
    session_id = request["session_id"]
    pid = request.get("pid")
    webrtc_enabled = request.get("webrtc_enabled", False)

    if webrtc_enabled:
        # Close WebRTC demo session
        success = await stop_webrtc_demo_session(session_id)
        return {"success": success, "session_type": "webrtc"}


@router.post("/gym_offer")
async def gym_offer(request: Request):
    """
    WebRTC offer for gymnasium environment streaming.
    Body JSON: {sdp, type, session_id, experiment_id, environment_id}
    """
    params = await request.json()

    try:
        client_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        session_id = params["session_id"]
        experiment_id = params["experiment_id"]
        environment_id = params["environment_id"]
    except KeyError as e:
        raise HTTPException(400, detail=f"Missing required parameter: {e}")

    exp = await db_handler.get_single_entry(database, Experiment, key=15)
    if exp is None:
        raise HTTPException(404, detail="Experiment not found")

    db_env = await db_handler.get_single_entry(database, Environment, key=12)
    if db_env is None:
        raise HTTPException(404, detail="Environment not found")

    ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

    # Try to create gymnasium environment track, fallback to test track
    try:

        gym_track = GymEnvironmentTrack(session_id=session_id, exp=exp, db_env=db_env, seed=42)
        pc.addTrack(gym_track)
        print(f"Created gymnasium track for {db_env.registration_id}")

        # Store for control message handling
        webrtc_demo_session.gym_sessions[session_id] = gym_track

    except Exception as e:
        print(f"Failed to create gymnasium track: {e}")
        print("Falling back to SimpleTestTrack...")

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str):
                if message.startswith("ping"):
                    channel.send("pong" + message[4:])
                else:
                    # Handle control messages - forward to gymnasium track
                    print(f"Control message: {message}")
                    if session_id in webrtc_demo_session.gym_sessions:
                        webrtc_demo_session.gym_sessions[session_id].handle_control_message(message)

    @pc.on("connectionstatechange")
    async def on_conn_state():
        if pc.connectionState == "failed":
            await pc.close()
            webrtc_demo_session.pcs.discard(pc)
            if session_id in webrtc_demo_session.gym_sessions:
                del webrtc_demo_session.gym_sessions[session_id]

    await pc.setRemoteDescription(client_offer)

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "session_id": session_id})


@router.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in webrtc_demo_session.pcs]
    await asyncio.gather(*coros)
    webrtc_demo_session.pcs.clear()


class FeedbackItem(BaseModel):
    """Model for feedback items."""

    feedback_type: str
    trajectory_id: int
    value: float
    metadata: dict[str, Any] = {}


@router.post("/train_iteration", response_model=dict[str, Any])
async def train_iteration(request: Request, background_tasks: BackgroundTasks):
    """
    Performs a training iteration for the given experiment: Trains the reward model with the provided feedback,
    generates new episodes, and then updates the experiment in the database. Returns the uncertainty and average
    predicted reward.

    Args:
        request: The HTTP request

    Returns:
        JSON response with submission status
    """
    # Get session_id, experiment_id, and phase from query parameters
    session_id = request.query_params.get("session_id", None)
    experiment_id = request.query_params.get("experiment_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if not session_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing session_id parameter"})

    if not experiment_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing experiment_id parameter"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        # Start simulated training
        simulation_key = f"{session_id}_{phase}"

        # Get the next training step number (increment from previous phases)
        training_step = 1
        for key, data in simulation_store.items():
            if key.startswith(f"{session_id}_") and "final_training_step" in data:
                training_step = max(training_step, data["final_training_step"] + 1)

        simulation_store[simulation_key] = {
            "status": "training",
            "start_time": time.time(),
            "training_step": training_step,
            "uncertainty": random.uniform(0.1, 0.5),
            "avg_reward": random.uniform(0.3, 0.8),
            "progress": 0.0,
        }

        return JSONResponse(
            content={
                "phaseStatus": "training_started",
                "message": "Training iteration started successfully",
                "phaseTrainingStep": training_step,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
                "session_id": session_id,
                "num_feedback": 0,
            }
        )

    # Original training logic for non-simulation mode
    exp = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "No experiment found"})

    # Check if the feedback translator is initialized
    translator: FeedbackTranslator = request.app.state.feedback_translator

    # Process current feedback from the translator
    translator.process()

    # Load feedback from the translator's logger
    in_feedback = translator.logger.read()

    if not in_feedback:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No feedback available for training"})

    try:
        # Create reward model handler with phase-specific directory
        handler = RewardModelHandler(exp, session_id, phase)

        # Submit feedback for training (this starts the background process)
        submission_status = handler.submit_feedback(in_feedback)

        if submission_status["status"] == "error":
            return JSONResponse(
                status_code=500,
                content={
                    "phaseStatus": "error",
                    "message": submission_status["message"],
                    "phaseTrainingStep": 0,
                    "phaseUncertainty": 0.0,
                    "phaseReward": 0.0,
                },
            )

        # Return immediate response indicating training has started
        return JSONResponse(
            content={
                "phaseStatus": "training_started",
                "message": "Training iteration started successfully",
                "phaseTrainingStep": 0,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
                "session_id": session_id,
                "num_feedback": len(in_feedback),
            }
        )

    except Exception as e:
        import traceback

        error_msg = f"Error starting training iteration: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        return JSONResponse(
            status_code=500,
            content={
                "phaseStatus": "error",
                "message": f"Failed to start training: {str(e)}",
                "phaseTrainingStep": 0,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
            },
        )


@router.get("/get_training_status", response_model=dict[str, Any])
async def get_training_status(request: Request):
    """
    Returns the training status of the current training iteration.

    Args:
        request: The HTTP request

    Returns:
        Dictionary with training status
    """
    session_id = request.query_params.get("session_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if session_id is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No session id given"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        simulation_key = f"{session_id}_{phase}"
        if simulation_key in simulation_store:
            sim_data = simulation_store[simulation_key]
            elapsed_time = time.time() - sim_data["start_time"]

            # Calculate progress (complete after 5 seconds)
            progress = min(elapsed_time / 5.0, 1.0)
            sim_data["progress"] = progress
            # Keep the training step constant during the 5-second simulation

            if progress >= 1.0:
                sim_data["status"] = "completed"

            return JSONResponse(
                content={
                    "status": sim_data["status"],
                    "message": ("Training in progress" if sim_data["status"] == "training" else "Training completed"),
                    "progress": progress,
                    "training_step": sim_data["training_step"],
                }
            )
        else:
            return JSONResponse(
                content={
                    "status": "idle",
                    "message": "No training in progress",
                }
            )

    # Get handler from cache or create a new one with phase
    handler = RewardModelHandler(None, session_id, phase)

    # Get training status
    status = handler.get_training_status()

    return status


@router.get("/get_training_results", response_model=dict[str, Any])
async def get_training_results(request: Request):
    """
    Returns the training results of the current training iteration:
    uncertainty and average predicted reward.

    Args:
        request: The HTTP request

    Returns:
        Dictionary with training results
    """
    session_id = request.query_params.get("session_id", None)
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    if session_id is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No session id given"})

    # Check if we should simulate training
    simulate_training = os.environ.get("SIMULATE_TRAINING", "false").lower() == "true"

    if simulate_training:
        simulation_key = f"{session_id}_{phase}"
        if simulation_key in simulation_store:
            sim_data = simulation_store[simulation_key]
            elapsed_time = time.time() - sim_data["start_time"]

            # Calculate progress (complete after 5 seconds)
            progress = min(elapsed_time / 5.0, 1.0)
            training_complete = progress >= 1.0

            # Update simulation data
            sim_data["progress"] = progress
            # Keep the training step constant during the 5-second simulation

            if training_complete:
                sim_data["status"] = "completed"
                # Store final training step for next phase increment
                final_step_key = f"{session_id}_final"
                simulation_store[final_step_key] = {"final_training_step": sim_data["training_step"]}
                # Clean up simulation data after completion
                del simulation_store[simulation_key]

            return JSONResponse(
                content={
                    "phaseStatus": "completed" if training_complete else "training",
                    "message": ("Training completed successfully" if training_complete else "Training in progress"),
                    "phaseTrainingStep": sim_data["training_step"],
                    "phaseUncertainty": sim_data["uncertainty"],
                    "phaseReward": sim_data["avg_reward"],
                    "trainingComplete": training_complete,
                    "modelPath": "",
                    "metrics": {
                        "training_step": sim_data["training_step"],
                        "progress": progress,
                        "simulated": True,
                    },
                }
            )
        else:
            return JSONResponse(
                content={
                    "phaseStatus": "idle",
                    "message": "No training data available",
                    "phaseTrainingStep": 0,
                    "phaseUncertainty": 0.0,
                    "phaseReward": 0.0,
                    "trainingComplete": False,
                    "modelPath": "",
                    "metrics": {},
                }
            )

    handler = RewardModelHandler(None, session_id, phase)

    # Get training results
    results = handler.get_training_results()

    # Transform results to match the expected frontend format
    if "status" not in results or results["status"] != "error":
        # Check if training is complete
        training_complete = results.get("training_complete", False)

        return JSONResponse(
            content={
                "phaseStatus": "completed" if training_complete else "training",
                "message": "Training completed successfully" if training_complete else "Training in progress",
                "phaseTrainingStep": results.get("metrics", {}).get("training_step", 0),
                "phaseUncertainty": results.get("uncertainty", 0.0),
                "phaseReward": results.get("avg_predicted_reward", 0.0),
                "trainingComplete": training_complete,
                "modelPath": results.get("trained_model_path", ""),
                "metrics": results.get("metrics", {}),
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "phaseStatus": "error",
                "message": results.get("message", "Training failed"),
                "phaseTrainingStep": 0,
                "phaseUncertainty": 0.0,
                "phaseReward": 0.0,
            },
        )


@router.post("/collect_initial_episodes", response_model=dict[str, Any])
async def collect_initial_episodes(request: Request):
    """
    Collects initial episodes using a random policy.

    Args:
        request: The HTTP request

    Returns:
        Dictionary with collected episodes
    """
    session_id = request.query_params.get("session_id", None)
    experiment_id = request.query_params.get("experiment_id", None)
    num_episodes = int(request.query_params.get("num_episodes", "10"))

    if not session_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing session_id parameter"})

    if not experiment_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing experiment_id parameter"})

    # Get experiment from database

    exp = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "No experiment found"})

    # Get phase from query parameters
    phase = int(request.query_params.get("phase", "0"))  # Default to phase 0

    # Get or create handler instance with phase
    handler = RewardModelHandler(exp, session_id, phase)

    # Collect initial episodes
    result = handler.initial_episode_collection(num_episodes)

    return result

import asyncio
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import httpx

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCIceServer
from databases import Database
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

from rlhfblender.data_collection import webrtc_demo_session
from rlhfblender.data_collection.demo_session import (
    create_new_session,
    demo_perform_step,
)
from rlhfblender.data_collection.webrtc_demo_session import GymEnvironmentTrack
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.global_models import Environment, Experiment
from rlhfblender.projections.inverse_state_projection_handler import InverseStateProjectionHandler
from rlhfblender.projections.projection_handler import ProjectionHandler
from rlhfblender.utils.data_generation import encode_video

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

router = APIRouter(prefix="/demo_generation")

# Global cache for ICE server credentials
_ice_server_cache: Dict[str, Any] = {
    "servers": None,
    "credential": None,
    "expires_at": 0
}


def _sanitize_component(value: Optional[Any]) -> str:
    """Create a filesystem-friendly identifier component."""
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    for token in ("/", "\\", ":", " "):
        text = text.replace(token, "_")
    return text


def _build_projection_output_dir(environment_id: str, experiment_id: Optional[int], checkpoint: Optional[int]) -> Path:
    env_component = _sanitize_component(environment_id)
    exp_component = f"exp-{experiment_id}" if experiment_id is not None else "exp-unknown"
    checkpoint_component = f"checkpoint-{checkpoint}" if checkpoint is not None else "checkpoint-unset"
    output_dir = Path("data") / "saved_projections" / "generated_demos" / env_component / exp_component / checkpoint_component
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_joint_projection_metadata(
    environment_id: str,
    experiment_id: Optional[int],
    projection_method: str
) -> tuple[Optional[Path], Optional[InverseStateProjectionHandler], str, Optional[Path]]:
    """
    Find the latest joint projection metadata for the given env/exp and return:
      (metadata_path, state_handler, effective_method, state_model_path)

    - Searches both saved_projections/joint_obs_state and saved_projections/joint
    - Falls back to PCA if the requested method is missing
    - Resolves and loads the inverse state model if available, preferring the path embedded
      in metadata; if not there, globs for a *_state_model.pkl with matching prefix.
    """
    if experiment_id is None:
        return None, None, projection_method, None

    env_component = _sanitize_component(environment_id)
    exp_component = _sanitize_component(experiment_id)

    joint_state_dir = Path("data") / "saved_projections" / "joint_obs_state"
    joint_dir = Path("data") / "saved_projections" / "joint"

    def _latest(globs: list[Path]) -> Optional[Path]:
        globs = [p for p in globs if p.exists()]
        return max(globs, key=lambda p: p.stat().st_mtime) if globs else None

    def _find_meta(method: str) -> Optional[Path]:
        pattern_state = f"{env_component}_{exp_component}_joint_obs_state_{method}_*_metadata.json"
        pattern_joint = f"{env_component}_{exp_component}_joint_{method}_*_metadata.json"
        candidates: list[Path] = []
        if joint_state_dir.exists():
            candidates.extend(joint_state_dir.glob(pattern_state))
        if joint_dir.exists():
            candidates.extend(joint_dir.glob(pattern_joint))
        return _latest(candidates)

    # 1) try requested method
    effective_method = projection_method or "PCA"
    meta = _find_meta(effective_method)

    # 2) fallback to PCA if needed
    if meta is None and effective_method != "PCA":
        effective_method = "PCA"
        meta = _find_meta(effective_method)

    # Early out if still nothing
    if meta is None:
        return None, None, effective_method, None

    # 3) try to resolve a state model path
    state_model_path: Optional[Path] = None
    try:
        with meta.open("r", encoding="utf-8") as f:
            md = json.load(f)
        if "state_model_path" in md and md["state_model_path"]:
            cand = Path(md["state_model_path"])
            if cand.exists():
                state_model_path = cand
    except Exception:
        pass

    # 4) if metadata didn't have it (or file is missing), glob a sensible fallback
    if state_model_path is None:
        # canonical prefix used across the codebase
        patt_state_model = (
            f"{env_component}_{exp_component}_joint_obs_state_{effective_method}_*_state_model.pkl"
        )
        candidates: list[Path] = []
        if joint_state_dir.exists():
            candidates.extend(joint_state_dir.glob(patt_state_model))
        state_model_path = _latest(candidates)

    # 5) create handler if we found a model
    state_handler: Optional[InverseStateProjectionHandler] = None
    if state_model_path and state_model_path.exists():
        try:
            state_handler = InverseStateProjectionHandler()
            state_handler.load_model(str(state_model_path))
        except Exception as e:
            print(f"Warning: failed to load inverse state model: {e}")
            state_handler = None

    return meta, state_handler, effective_method, state_model_path



def _load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _compute_episode_indices(dones: np.ndarray) -> List[int]:
    episode_indices: List[int] = []
    current_episode = 0
    for flag in dones:
        episode_indices.append(current_episode)
        if bool(flag):
            current_episode += 1
    return episode_indices


def _prepare_demo_artifacts(
    track: GymEnvironmentTrack,
    demo_path: Path,
    *,
    projection_method_override: Optional[str] = None,
    projection_props_override: Optional[dict] = None,
) -> Dict[str, Any]:
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    metadata_path = demo_path.with_suffix(".json")
    metadata_dict: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata_dict = json.load(meta_file)

    demo_arrays = _load_npz_arrays(demo_path)

    actions = np.array(demo_arrays.get("actions", []))
    rewards = np.array(demo_arrays.get("rewards", []))
    dones = np.array(demo_arrays.get("dones", []))
    episode_steps_raw = np.array(demo_arrays.get("episode_steps", np.arange(len(actions))))
    obs_array = demo_arrays.get("obs", np.array([]))

    if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
        obs_array = np.stack(obs_array.astype(np.float32)) if len(obs_array) else np.array([], dtype=np.float32)
    else:
        obs_array = np.array(obs_array)

    if obs_array.ndim == 0 and actions.shape[0] > 0:
        obs_array = np.repeat(obs_array[None], actions.shape[0], axis=0)

    num_steps = actions.shape[0]

    if obs_array.shape[0] > num_steps:
        obs_array = obs_array[:num_steps]
    elif obs_array.shape[0] < num_steps:
        if obs_array.shape[0] == 0:
            obs_array = np.zeros((num_steps, 1), dtype=np.float32)
        else:
            last_frame = obs_array[-1]
            pad = np.repeat(last_frame[None, ...], num_steps - obs_array.shape[0], axis=0)
            obs_array = np.concatenate([obs_array, pad], axis=0)

    if rewards.shape[0] > num_steps:
        rewards = rewards[:num_steps]
    elif rewards.shape[0] < num_steps:
        rewards = np.pad(rewards, (0, num_steps - rewards.shape[0]), mode="constant")

    if dones.shape[0] > num_steps:
        dones = dones[:num_steps]
    elif dones.shape[0] < num_steps:
        dones = np.pad(dones, (0, num_steps - dones.shape[0]), mode="constant")

    if episode_steps_raw.shape[0] >= num_steps:
        episode_steps = episode_steps_raw[:num_steps]
    elif episode_steps_raw.shape[0] > 0:
        start_idx = int(episode_steps_raw[-1]) + 1
        extra = np.arange(start_idx, start_idx + (num_steps - episode_steps_raw.shape[0]))
        episode_steps = np.concatenate([episode_steps_raw, extra])
    else:
        episode_steps = np.arange(num_steps)

    # Normalize observation shape for projection handler
    obs_array = np.array(obs_array, dtype=np.float32)
    if obs_array.ndim > 2:
        obs_array = obs_array.reshape(obs_array.shape[0], -1)
    elif obs_array.ndim == 1:
        obs_array = obs_array.reshape(-1, 1)

    # Prepare renders and encode video if available
    video_path: Optional[Path] = None
    renders = demo_arrays.get("renders")
    if isinstance(renders, np.ndarray) and renders.ndim == 4 and renders.shape[0] > 0:
        video_base = demo_path.with_suffix("")
        candidate_path = Path(f"{video_base}.mp4")
        if not candidate_path.exists():
            render_frames = renders
            if render_frames.dtype != np.uint8:
                if render_frames.size and render_frames.max() <= 1.0:
                    render_frames = (render_frames * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    render_frames = render_frames.clip(0, 255).astype(np.uint8)
            encode_video(render_frames, str(video_base))
        video_path = candidate_path

    projection_method = (
        projection_method_override
        or track.projection_method
        or metadata_dict.get("projection_method")
        or "PCA"
    )
    projection_props = projection_props_override or track.projection_props

    meta_path, _, projection_method, _ = _find_joint_projection_metadata(
        track.environment_id, track.experiment_id, projection_method
    )

    joint_metadata: Dict[str, Any] = {}
    if meta_path and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as joint_file:
            joint_metadata = json.load(joint_file)
            if projection_props is None:
                projection_props = joint_metadata.get("projection_props")

    projection_array = np.zeros((0, 2), dtype=np.float32)
    if meta_path is not None:
        sequence_length = joint_metadata.get("sequence_length", 1)
        handler = ProjectionHandler(
            projection_method=projection_method,
            projection_props=projection_props,
            joint_projection_path=str(meta_path),
        )
        try:
            projection_raw = handler.fit(
                obs_array,
                sequence_length=sequence_length,
                step_range=None,
                episode_indices=None,
                actions=None,
                suffix=f"user_demo_{track.session_id}",
            )
            projection_array = np.array(projection_raw, dtype=np.float32)
        except Exception as exc:
            print(f"Failed to project demo {demo_path.name}: {exc}")
            projection_array = np.zeros((0, 2), dtype=np.float32)


    # Fallback: if we could not load a joint projection, generate a fresh 2D projection directly
    if (projection_array.size == 0 or projection_array.shape[0] != obs_array.shape[0]) and obs_array.size > 0:
        try:
            fallback_method = projection_method or "PCA"
            fallback_props = projection_props or {}
            fallback_handler = ProjectionHandler(
                projection_method=fallback_method,
                projection_props=fallback_props,
            )
            projection_raw = fallback_handler.fit(
                obs_array,
                sequence_length=1,
                step_range=None,
                episode_indices=None,
                actions=None,
                suffix=f"user_demo_{track.session_id}_local",
            )
            projection_array = np.array(projection_raw, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"Fallback projection failed for demo {demo_path.name}: {exc}")
            projection_array = np.zeros((0, 2), dtype=np.float32)

    episode_indices = _compute_episode_indices(dones)
    projection_output_dir = _build_projection_output_dir(track.environment_id, track.experiment_id, track.current_checkpoint)
    projection_json_path = projection_output_dir / f"{demo_path.stem}.json"

    payload: Dict[str, Any] = {
        "demo_file": str(demo_path),
        "metadata_file": str(metadata_path) if metadata_path.exists() else None,
        "projection_file": str(projection_json_path),
        "projection_method": projection_method,
        "joint_metadata_path": str(meta_path) if meta_path else None,
        "projection": projection_array.tolist(),
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
        "episode_indices": episode_indices,
        "episode_steps": episode_steps.astype(int).tolist(),
        "video_path": str(video_path) if video_path else None,
        "total_reward": float(rewards.sum()) if rewards.size else 0.0,
        "num_steps": int(num_steps),
        "metadata": metadata_dict,
    }

    with projection_json_path.open("w", encoding="utf-8") as proj_file:
        json.dump(payload, proj_file, indent=2)

    return payload

async def create_expiring_turn_credential() -> Dict[str, Any]:
    """
    Create expiring TURN credentials using the Metered API.
    Returns the credential info including username, password, and apiKey.
    """
    secret_key = os.environ.get("METERED_SECRET_KEY")
    application_name = os.environ.get("METERED_APP_NAME")
    if not secret_key:
        raise HTTPException(500, detail="METERED_SECRET_KEY not configured")
    if not application_name:
        raise HTTPException(500, detail="METERED_APP_NAME not configured")
    
    # Create credential that expires in 4 hours (1800 seconds)
    url = f"https://{application_name}.metered.live/api/v1/turn/credential?secretKey={secret_key}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers={'Content-Type': 'application/json'},
            json={
                "expiryInSeconds": 1800,  # 30 mins
                "label": "rlhfblender-session"
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(500, detail=f"Failed to create TURN credential: {response.text}")
        
        return response.json()


async def get_cached_ice_servers() -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Get cached ICE servers or create new ones if cache is expired.
    Returns (ice_servers_data, credential) tuple.
    """
    current_time = time.time()
    
    # Check if cache is still valid (with 5min safety margin)
    if (_ice_server_cache["expires_at"] > current_time and 
        _ice_server_cache["servers"] is not None):
        print("Using cached ICE servers")
        return _ice_server_cache["servers"], _ice_server_cache["credential"]
    
    try:
        print("Creating new ICE server credentials")
        # Create new credentials
        credential = await create_expiring_turn_credential()
        ice_servers_data = await get_ice_servers_from_credential(credential["apiKey"])
        
        # Cache with expiry (25min to be safe, original is 30min)
        _ice_server_cache.update({
            "servers": ice_servers_data,
            "credential": credential,
            "expires_at": current_time + 1500  # 25 minutes
        })
        
        return ice_servers_data, credential
        
    except Exception as e:
        print(f"Failed to get ICE servers: {e}")
        # Return fallback
        fallback_servers = [{"urls": "stun:stun.l.google.com:19302"}]
        return fallback_servers, None


async def get_ice_servers_from_credential(api_key: str) -> list[dict[str, Any]]:
    """
    Fetch ICE servers using the API key from expiring credential.
    """
    application_name = os.environ.get("METERED_APP_NAME")
    if not application_name:
        raise HTTPException(500, detail="METERED_APP_NAME not configured")

    url = f"https://{application_name}.metered.live/api/v1/turn/credentials?apiKey={api_key}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code != 200:
            raise HTTPException(500, detail=f"Failed to fetch ICE servers: {response.text}")
        
        return response.json()


@router.get("/ice_servers")
async def ice_servers():
    """
    Return cached or newly created ICE servers for WebRTC (TURN/STUN).
    This lets the frontend fetch ICE config before creating RTCPeerConnection.
    """
    try:
        ice_servers_data, credential = await get_cached_ice_servers()
        return JSONResponse({
            "iceServers": ice_servers_data,
            "credentialExpiry": credential.get("expiryInSeconds", 1800) if credential else 1800
        })
    except Exception as e:
        # Fallback to public STUN to avoid total failure
        return JSONResponse({
            "iceServers": [{"urls": "stun:stun.l.google.com:19302"}],
            "error": str(e)
        })




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


async def stop_webrtc_demo_session(session_id: str) -> bool:
    """
    Helper function to stop a WebRTC demo session
    """
    try:
        if session_id in webrtc_demo_session.gym_sessions:
            # Save demo data before cleaning up
            track = webrtc_demo_session.gym_sessions[session_id]
            try:
                demo_number = track.demo_counter
                saved_path = track.save_demo_data(demo_number)
                if saved_path:
                    artifacts = _prepare_demo_artifacts(track, saved_path)
                    print(f"Successfully saved WebRTC demo data for session {session_id}")
                    print(f"Artifacts stored at {artifacts.get('projection_file')}")
                else:
                    print(f"No demo data recorded for session {session_id}")
            except Exception as e:
                print(f"Error saving WebRTC demo data: {e}")
            
            # Clean up the gym session
            track.stop()
            del webrtc_demo_session.gym_sessions[session_id]
        return True
    except Exception as e:
        print(f"Error stopping WebRTC demo session: {e}")
        return False


@router.post("/save_webrtc_demo")
async def save_webrtc_demo(request: Request):
    """
    Save the current WebRTC demo session data to file
    """
    request = await request.json()
    session_id = request["session_id"]
    projection_method = request.get("projection_method")
    projection_props = request.get("projection_props")
    checkpoint = request.get("checkpoint")

    try:
        if session_id not in webrtc_demo_session.gym_sessions:
            return {"success": False, "message": "Session not found"}
            
        track = webrtc_demo_session.gym_sessions[session_id]

        if checkpoint is not None:
            try:
                track.current_checkpoint = int(checkpoint)
            except (TypeError, ValueError):
                track.current_checkpoint = None

        if projection_method:
            track.projection_method = projection_method
        if projection_props:
            track.projection_props = projection_props

        demo_number = request.get("demo_number")
        if demo_number is None:
            demo_number = track.demo_counter

        try:
            demo_number = int(demo_number)
        except (TypeError, ValueError):
            demo_number = track.demo_counter

        saved_path = track.save_demo_data(demo_number)

        if not saved_path:
            return {"success": False, "message": "Failed to save demo data"}

        artifacts = _prepare_demo_artifacts(
            track,
            saved_path,
            projection_method_override=projection_method,
            projection_props_override=projection_props,
        )

        return {
            "success": True,
            "message": f"Demo saved as {saved_path.name}",
            "demo_number": demo_number,
            "file_path": str(saved_path),
            "artifacts": artifacts,
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@router.post("/gym_offer")
async def gym_offer(request: Request):
    """
    WebRTC offer for gymnasium environment streaming.
    Body JSON: {sdp, type, session_id, experiment_id, environment_id, coordinate?, episode_num?, step?}
    """
    params = await request.json()

    try:
        client_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        session_id = params["session_id"]
        experiment_id = params["experiment_id"]
        environment_id = params["environment_id"]

        # Optional parameters for state loading
        coordinate = params.get("coordinate")  # [x, y] coordinate pair
        episode_num = params.get("episode_num")  # Episode number for saved state loading
        step = params.get("step")  # Step number for saved state loading
        checkpoint = params.get("checkpoint")
        projection_method = params.get("projection_method")
        projection_props = params.get("projection_props")

        print("Received WebRTC offer for session:", session_id, "experiment:", experiment_id, " environment:", environment_id)
        if coordinate:
            print(f"  Will load state from coordinate: {coordinate}")
        elif episode_num is not None and step is not None:
            print(f"  Will load saved state from checkpoint {checkpoint}, episode {episode_num}, step {step}")
    except KeyError as e:
        raise HTTPException(400, detail=f"Missing required parameter: {e}")

    exp = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        raise HTTPException(404, detail="Experiment not found")

    db_env = await db_handler.get_single_entry(database, Environment, key=environment_id, key_column="registration_id")
    if db_env is None:
        raise HTTPException(404, detail="Environment not found")

    # Get ICE servers using caching
    ice_servers_data, credential = await get_cached_ice_servers()
    
    # Convert to RTCIceServer objects for WebRTC peer connection
    ice_servers = []
    for server in ice_servers_data:
        ice_servers.append(RTCIceServer(
            urls=[server["urls"]],
            username=server.get("username"),
            credential=server.get("credential")
        ))
    
    print(f"Using {len(ice_servers)} ICE servers for WebRTC connection")

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

    webrtc_demo_session.pcs.add(pc)

    # Try to create or reuse gymnasium environment track for this session
    try:
        print(f"Creating track for {db_env.registration_id}")

        # Load initial state if coordinate or env_state is provided
        initial_state = None
        if coordinate:
            try:
                meta_path, state_handler, method, model_path = _find_joint_projection_metadata(
                    environment_id, experiment_id, projection_method or "PCA"
                )
                if not state_handler:
                    raise HTTPException(404, detail="No inverse state projection model found for this experiment/environment")

                print(f"Loading state from coordinate {coordinate} using model {model_path}")
                predicted_states = state_handler.predict(np.array([coordinate], dtype=np.float32))

                initial_state = predicted_states[0]
                print("Successfullxwxy predicted initial state from coordinate")

            except Exception as e:
                print(f"Failed to load initial state from coordinate: {e}")
                initial_state = None
        elif episode_num is not None and step is not None:
            try:
                # Load env_state from saved episode data
                import numpy as np

                # Construct the episode file path
                episode_file_path = (
                    f"data/env_states/{environment_id}/{environment_id}_{experiment_id}_{checkpoint}/env_states_{episode_num}.npy"
                )

                # Load the episode data
                env_states = np.load(episode_file_path, allow_pickle=True)

                # Check if the requested step exists
                if step >= len(env_states):
                    raise Exception(f"Step {step} not found in episode {episode_num} (max step: {len(env_states)-1})")

                # Get the env_state for the requested step
                env_state = env_states[step][0]["state"]

                if env_state is None:
                    raise Exception(f"env_state is None for episode {episode_num}, step {step}")

                # Handle numpy objects
                initial_state = env_state.item() if hasattr(env_state, "item") else env_state
                print(f"Successfully loaded saved state from episode {episode_num}, step {step}")

            except FileNotFoundError:
                print(f"Episode file not found for episode {episode_num}")
                initial_state = None
            except Exception as e:
                print(f"Failed to load saved state: {e}")
                initial_state = None

        # Session cache with TTL
        SESSION_TTL_SECONDS = 30 * 60
        import time as _time

        # Lazy cleanup of expired sessions
        try:
            to_delete = []
            for sid, t in list(webrtc_demo_session.gym_sessions.items()):
                if getattr(t, 'stopped', False):
                    to_delete.append(sid)
                    continue
                last_access = getattr(t, 'last_access', 0)
                if last_access and (_time.time() - last_access) > SESSION_TTL_SECONDS:
                    try:
                        t.stop()
                    except Exception:
                        pass
                    to_delete.append(sid)
            for sid in to_delete:
                del webrtc_demo_session.gym_sessions[sid]
        except Exception as e:
            print(f"Session cleanup warning: {e}")

        # Reuse existing track if available and not expired
        if session_id in webrtc_demo_session.gym_sessions:
            gym_track = webrtc_demo_session.gym_sessions[session_id]
            print(f"Reusing existing environment for session {session_id}")
            gym_track.touch()
            if checkpoint is not None:
                try:
                    gym_track.current_checkpoint = int(checkpoint)
                except (TypeError, ValueError):
                    gym_track.current_checkpoint = None
            if projection_method:
                gym_track.projection_method = projection_method
            if projection_props:
                gym_track.projection_props = projection_props
        else:
            gym_track = GymEnvironmentTrack(
                session_id=session_id,
                exp=exp,
                db_env=db_env,
                seed=42,
                initial_state=initial_state,
                target_width=480,
                target_height=360,
                target_fps=15,
            )
            # Small delay to ensure track is properly initialized
            await asyncio.sleep(0.1)
            if checkpoint is not None:
                try:
                    gym_track.current_checkpoint = int(checkpoint)
                except (TypeError, ValueError):
                    gym_track.current_checkpoint = None
            if projection_method:
                gym_track.projection_method = projection_method
            if projection_props:
                gym_track.projection_props = projection_props
            webrtc_demo_session.gym_sessions[session_id] = gym_track
            print(f"Created and cached new environment for session {session_id}")

        # Use MediaRelay to subscribe a per-connection track to the shared session track
        local_track = webrtc_demo_session.relay.subscribe(gym_track)
        sender = pc.addTrack(local_track)
        # Prefer H264 when available (often hardware-accelerated in browsers)
        try:
            from aiortc.rtcrtpsender import RTCRtpSender
            caps = RTCRtpSender.getCapabilities("video")
            preferred = [c for c in caps.codecs if getattr(c, 'mimeType', '').lower() == 'video/h264']
            fallback = [c for c in caps.codecs if getattr(c, 'mimeType', '').lower() != 'video/h264']
            for transceiver in pc.getTransceivers():
                if transceiver.kind == 'video' and hasattr(transceiver, 'setCodecPreferences'):
                    transceiver.setCodecPreferences(preferred + fallback)
        except Exception as e:
            print(f"Codec preference setup skipped: {e}")
        print(f"Track created successfully")

        # Store data channel handler mapping
        print(f"Session {session_id} ready")

    except Exception as e:
        print(f"Failed to create gymnasium track: {e}")
        import traceback

        traceback.print_exc()
        # Continue without raising - let WebRTC try to work without the track
        print("Continuing without gymnasium track - WebRTC will work with limited functionality")

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"Data channel received: {channel.label}")

        # Store reference to the data channel for this session
        webrtc_demo_session.data_channels[session_id] = channel

        @channel.on("open")
        def on_open():
            print(f"Data channel {channel.label} opened")
            try:
                channel.send("server_ready")
            except Exception as e:
                print(f"Error sending server_ready: {e}")

        @channel.on("close")
        def on_close():
            print(f"Data channel {channel.label} closed")
            if session_id in webrtc_demo_session.data_channels:
                del webrtc_demo_session.data_channels[session_id]

        @channel.on("message")
        def on_message(message):
            # Handle both string and bytes messages
            try:
                if isinstance(message, bytes):
                    message_str = message.decode("utf-8")
                elif isinstance(message, str):
                    message_str = message
                else:
                    print(f"Unexpected message type: {type(message)}")
                    return

                # Handle ping/pong for connection testing
                if message_str.startswith("ping"):
                    response = "pong" + message_str[4:]
                    channel.send(response)
                    return

                # Forward control messages to gymnasium track
                if session_id in webrtc_demo_session.gym_sessions:
                    webrtc_demo_session.gym_sessions[session_id].handle_control_message(message_str)
                else:
                    print(f"Session {session_id} not found")

            except Exception as e:
                print(f"Error processing message: {e}")

    @pc.on("connectionstatechange")
    async def on_conn_state():
        print(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed":
            print(f"Connection failed for session {session_id}, cleaning up...")
            await pc.close()
            webrtc_demo_session.pcs.discard(pc)
            if session_id in webrtc_demo_session.gym_sessions:
                # Clean up the track properly
                try:
                    track = webrtc_demo_session.gym_sessions[session_id]
                    track.stop()
                except Exception as e:
                    print(f"Error stopping track: {e}")
                del webrtc_demo_session.gym_sessions[session_id]
                print(f"Cleaned up session {session_id}")
        elif pc.connectionState == "connected":
            print(f"Successfully connected session {session_id}")

    await pc.setRemoteDescription(client_offer)

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp, 
        "type": pc.localDescription.type, 
        "session_id": session_id,
        "iceServers": ice_servers_data,
        "credentialExpiry": credential.get("expiryInSeconds", 1800) if credential else 1800
    })


@router.post("/coordinate_to_render")
async def coordinate_to_render(request: Request):
    """
    Convert 2D coordinates to environment state using inverse state projection,
    load that state into an environment, and return a rendered frame.

    This enables generating novel states from clicking on projection visualizations.
    """
    try:
        request_data = await request.json()

        # Extract parameters
        coordinates = np.array(request_data.get("coordinates"))  # Shape: (N, 2) or (2,)
        env_id = request_data.get("env_id")
        exp_id = request_data.get("exp_id")

        if coordinates is None:
            raise HTTPException(status_code=400, detail="coordinates are required")
        if env_id is None:
            raise HTTPException(status_code=400, detail="env_id is required")
        if exp_id is None:
            raise HTTPException(status_code=400, detail="exp_id is required")

        # Ensure coordinates are 2D array
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        # Get environment and experiment info from database
        exp = await db_handler.get_single_entry(database, Experiment, key=exp_id)
        db_env = await db_handler.get_single_entry(database, Environment, key=env_id, key_column="registration_id")

        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
        if db_env is None:
            raise HTTPException(status_code=404, detail=f"Environment {env_id} not found")

        meta_path, state_handler, method, model_path = _find_joint_projection_metadata(
            env_id, exp_id, "PCA"
        )
        if not state_handler:
            raise HTTPException(status_code=404, detail="No inverse state projection model found")

        predicted_states = state_handler.predict(coordinates.astype(np.float32))

        if not predicted_states:
            raise HTTPException(status_code=500, detail="Failed to predict states from coordinates")

        # Use the first predicted state
        target_state = predicted_states[0]

        print(f"Predicted state from coordinates {coordinates.tolist()}: {target_state}")

        # Create render from state using helper function
        try:
            render_frame = webrtc_demo_session.create_render_from_state(exp, db_env, target_state)

            # Convert numpy array to base64 encoded image
            if isinstance(render_frame, np.ndarray):
                # Convert to PIL Image
                if render_frame.dtype != np.uint8:
                    render_frame = (render_frame * 255).astype(np.uint8)

                image = Image.fromarray(render_frame)

                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()

                return JSONResponse(
                    {
                        "success": True,
                        "render_frame": img_str,
                        "coordinates": coordinates.tolist(),
                        "state_keys": list(target_state.keys()) if isinstance(target_state, dict) else None,
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Environment rendering failed")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load state or render: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/initialize_demo_from_coordinate")
async def initialize_demo_from_coordinate(request: Request):
    """
    Initialize a new demo session starting from a state predicted from 2D coordinates.
    Similar to initialize_demo_session but loads a specific state from coordinates.
    """
    try:
        request_data = await request.json()

        # Extract parameters
        joint_projection_path = request_data.get("joint_projection_path")
        coordinates = np.array(request_data.get("coordinates"))
        env_id = request_data.get("env_id")
        exp_id = request_data.get("exp_id")
        seed = request_data.get("seed", 42)
        session_id = request_data.get("session_id")

        if not all([joint_projection_path, coordinates is not None, env_id, exp_id, session_id]):
            raise HTTPException(status_code=400, detail="Missing required parameters")

        # Ensure coordinates are 2D array
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        # Load projection handler and predict state (same as above)
        projection_handler = ProjectionHandler(joint_projection_path=joint_projection_path)

        if not hasattr(projection_handler, "state_model_path") or projection_handler.state_model_path is None:
            raise HTTPException(status_code=400, detail="No inverse state projection model found in joint projection")

        state_handler = InverseStateProjectionHandler()
        state_handler.load_model(projection_handler.state_model_path)
        predicted_states = state_handler.predict(coordinates)

        print("PREDICTED STATES:", predicted_states)

        if not predicted_states:
            raise HTTPException(status_code=500, detail="Failed to predict states from coordinates")

        target_state = predicted_states[0]

        # Get database entries
        exp = await db_handler.get_single_entry(database, Experiment, key=exp_id)
        db_env = await db_handler.get_single_entry(database, Environment, key=env_id, key_column="registration_id")

        if exp is None or db_env is None:
            raise HTTPException(status_code=404, detail="Experiment or environment not found")

        # Create WebRTC demo session with the predicted state as initial_state
        try:
            gym_track = GymEnvironmentTrack(
                session_id=session_id, exp=exp, db_env=db_env, seed=int(seed), initial_state=target_state
            )

            # Store the track in the global sessions dict
            webrtc_demo_session.gym_sessions[session_id] = gym_track

            success = True
            demo_number = 0  # WebRTC sessions don't use demo numbers the same way
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create demo session: {str(e)}")

        return JSONResponse(
            {
                "success": success,
                "session_id": session_id,
                "demo_number": demo_number,
                "action_space": db_env.action_space_info,
                "coordinates": coordinates.tolist(),
                "state_keys": list(target_state.keys()) if isinstance(target_state, dict) else None,
                "message": "WebRTC demo session created with predicted state. Use WebRTC to connect and start demo.",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.on_event("shutdown")
async def on_shutdown():
    """Handle router shutdown - close all WebRTC connections"""
    coros = [pc.close() for pc in webrtc_demo_session.pcs]
    await asyncio.gather(*coros)
    webrtc_demo_session.pcs.clear()

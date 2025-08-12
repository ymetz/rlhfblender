import asyncio
import base64
import os
from io import BytesIO
from typing import List, Dict, Any
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

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

router = APIRouter(prefix="/demo_generation")


async def create_expiring_turn_credential() -> Dict[str, Any]:
    """
    Create expiring TURN credentials using the Metered API.
    Returns the credential info including username, password, and apiKey.
    """
    secret_key = os.environ.get("METERED_SECRET_KEY")
    if not secret_key:
        raise HTTPException(500, detail="METERED_SECRET_KEY not configured")
    
    # Create credential that expires in 4 hours (1800 seconds)
    url = f"https://mla2.metered.live/api/v1/turn/credential?secretKey={secret_key}"
    
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


async def get_ice_servers_from_credential(api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch ICE servers using the API key from expiring credential.
    """
    url = f"https://mla2.metered.live/api/v1/turn/credentials?apiKey={api_key}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code != 200:
            raise HTTPException(500, detail=f"Failed to fetch ICE servers: {response.text}")
        
        return response.json()


@router.get("/webrtc_config")
async def get_webrtc_config():
    """
    Get WebRTC ICE server configuration with expiring TURN credentials.
    This endpoint should be called by the frontend before establishing WebRTC connections.
    """
    try:
        # Create expiring credential
        credential = await create_expiring_turn_credential()
        
        # Get ICE servers using the API key
        ice_servers = await get_ice_servers_from_credential(credential["apiKey"])
        
        # Add STUN server as fallback
        stun_server_host = os.environ.get("WEBRTC_STUN_HOST", "stun.relay.metered.ca")
        ice_servers.insert(0, {
            "urls": f"stun:{stun_server_host}:80"
        })
        
        return JSONResponse({
            "iceServers": ice_servers,
            "credentialExpiry": credential["expiryInSeconds"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get WebRTC configuration: {str(e)}")


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
            # Clean up the gym session
            del webrtc_demo_session.gym_sessions[session_id]
        return True
    except Exception as e:
        print(f"Error stopping WebRTC demo session: {e}")
        return False


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

    # Get ICE servers using expiring credentials
    try:
        credential = await create_expiring_turn_credential()
        ice_servers_data = await get_ice_servers_from_credential(credential["apiKey"])
        
        # Convert to RTCIceServer objects
        ice_servers = []
        
        # Add STUN server
        stun_server_host = os.environ.get("WEBRTC_STUN_HOST", "stun.relay.metered.ca")
        ice_servers.append(RTCIceServer(urls=[f"stun:{stun_server_host}:80"]))
        
        # Add TURN servers from API response
        for server in ice_servers_data:
            ice_servers.append(RTCIceServer(
                urls=[server["urls"]],
                username=server.get("username"),
                credential=server.get("credential")
            ))
            
    except Exception as e:
        print(f"Warning: Failed to get expiring TURN credentials: {e}")
        # Fallback to basic STUN server only
        stun_server_host = os.environ.get("WEBRTC_STUN_HOST", "stun.l.google.com")
        ice_servers = [RTCIceServer(urls=[f"stun:{stun_server_host}:19302"])]

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

    webrtc_demo_session.pcs.add(pc)

    # Try to create gymnasium environment track
    try:
        print(f"Creating track for {db_env.registration_id}")

        # Load initial state if coordinate or env_state is provided
        initial_state = None
        if coordinate:
            try:
                import numpy as np

                from rlhfblender.projections.inverse_state_projection_handler import InverseStateProjectionHandler

                # Construct state model path from experiment info
                # Use the experiment's checkpoint list to get the min/max checkpoints
                checkpoint_list = exp.checkpoint_list if hasattr(exp, "checkpoint_list") and exp.checkpoint_list else []

                min_checkpoint = min(checkpoint_list)
                max_checkpoint = max(checkpoint_list)
                state_model_path = f"data/saved_projections/joint_obs_state/{environment_id}_{experiment_id}_joint_obs_state_PCA_{min_checkpoint}_{max_checkpoint}_state_model.pkl"

                print(f"Loading state from coordinate {coordinate} using model {state_model_path}")
                handler = InverseStateProjectionHandler()
                handler.load_model(state_model_path)

                # Predict state from coordinate
                predicted_states = handler.predict(np.array([coordinate]))
                initial_state = predicted_states[0]
                print("Successfully predicted initial state from coordinate")

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

        gym_track = GymEnvironmentTrack(session_id=session_id, exp=exp, db_env=db_env, seed=42, initial_state=initial_state)

        # Small delay to ensure track is properly initialized
        await asyncio.sleep(0.1)

        pc.addTrack(gym_track)
        print(f"Track created successfully")

        # Store for control message handling
        webrtc_demo_session.gym_sessions[session_id] = gym_track
        print(f"Session {session_id} stored")

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

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "session_id": session_id})


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

        # Construct paths from experiment info
        checkpoint_list = exp.checkpoint_list if hasattr(exp, "checkpoint_list") and exp.checkpoint_list else []
        min_checkpoint = min(checkpoint_list, key=int) if checkpoint_list else None
        max_checkpoint = max(checkpoint_list, key=int) if checkpoint_list else None
        state_model_path = f"data/saved_projections/joint_obs_state/{env_id}_{exp_id}_joint_obs_state_PCA_{min_checkpoint}_{max_checkpoint}_state_model.pkl"


        # Load inverse state projection model directly
        state_handler = InverseStateProjectionHandler()
        state_handler.load_model(state_model_path)

        # Predict states from coordinates
        predicted_states = state_handler.predict(coordinates)

        if not predicted_states:
            raise HTTPException(status_code=500, detail="Failed to predict states from coordinates")

        # Use the first predicted state
        target_state = predicted_states[0]

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

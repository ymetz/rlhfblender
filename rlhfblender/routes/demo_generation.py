import asyncio
import os

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCIceServer
from databases import Database
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from rlhfblender.data_collection import webrtc_demo_session
from rlhfblender.data_collection.demo_session import (
    create_new_session,
    demo_perform_step,
)
from rlhfblender.data_collection.webrtc_demo_session import GymEnvironmentTrack
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.global_models import Environment, Experiment

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

router = APIRouter(prefix="/demo_generation")


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
    Body JSON: {sdp, type, session_id, experiment_id, environment_id}
    """
    params = await request.json()

    try:
        client_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        session_id = params["session_id"]
        experiment_id = params["experiment_id"]
        environment_id = params["environment_id"]
        print("Received WebRTC offer for session:", session_id,
              "experiment:", experiment_id, " environment:", environment_id)
    except KeyError as e:
        raise HTTPException(400, detail=f"Missing required parameter: {e}")

    exp = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        raise HTTPException(404, detail="Experiment not found")

    db_env = await db_handler.get_single_entry(database, Environment, key=environment_id, key_column="registration_id")
    if db_env is None:
        raise HTTPException(404, detail="Environment not found")

    ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"])]

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

    webrtc_demo_session.pcs.add(pc)

    # Try to create gymnasium environment track
    try:
        print(f"Creating track for {db_env.registration_id}")
        gym_track = GymEnvironmentTrack(session_id=session_id, exp=exp, db_env=db_env, seed=42)
        
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
                    message_str = message.decode('utf-8')
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


@router.on_event("shutdown")
async def on_shutdown():
    """Handle router shutdown - close all WebRTC connections"""
    coros = [pc.close() for pc in webrtc_demo_session.pcs]
    await asyncio.gather(*coros)
    webrtc_demo_session.pcs.clear()

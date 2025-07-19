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
    except KeyError as e:
        raise HTTPException(400, detail=f"Missing required parameter: {e}")

    exp = await db_handler.get_single_entry(database, Experiment, key=experiment_id)
    if exp is None:
        raise HTTPException(404, detail="Experiment not found")

    db_env = await db_handler.get_single_entry(database, Environment, key=environment_id)
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
    """Handle router shutdown - close all WebRTC connections"""
    coros = [pc.close() for pc in webrtc_demo_session.pcs]
    await asyncio.gather(*coros)
    webrtc_demo_session.pcs.clear()

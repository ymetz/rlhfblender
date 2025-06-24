import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, VideoStreamTrack
from av import VideoFrame

from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.data_models.global_models import Environment, Experiment


# Setup debug logging
def setup_webrtc_logging():
    """Setup file-based logging for WebRTC demo sessions."""
    os.makedirs("logs/webrtc", exist_ok=True)

    # Create logger
    logger = logging.getLogger("webrtc_demo")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"logs/webrtc/webrtc_demo_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
webrtc_logger = setup_webrtc_logging()

# Global registry for active demo sessions
_demo_sessions: Dict[str, "WebRTCDemoSession"] = {}


class EnvironmentVideoTrack(VideoStreamTrack):
    """Minimal working video track."""
    
    def __init__(self, session_id: str):
        super().__init__()  # CRITICAL - must call parent init
        self.session_id = session_id
        self.counter = 0
        webrtc_logger.info(f"Created video track for {session_id}")
        
    async def recv(self):
        """Generate and return a video frame."""
        pts, time_base = await self.next_timestamp()
        
        # Log to confirm recv is being called
        if self.counter % 30 == 0:  # Log once per second
            webrtc_logger.info(f"recv() called {self.counter} times")
        
        # Create a simple test pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving elements to verify it's updating
        bar_pos = (self.counter * 5) % 640
        img[:, bar_pos:bar_pos+20, 0] = 255  # Red moving bar
        
        # Add counter in corner
        cv = self.counter % 255
        img[0:50, 0:50] = [cv, cv, cv]
        
        self.counter += 1
        
        # Create frame
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        
        return frame

class WebRTCDemoSession:
    """WebRTC-based demo session for real-time environment interaction."""

    def __init__(self, session_id: str, exp: Experiment, db_env: Environment, seed: int):
        self.session_id = session_id
        self.exp = exp
        self.db_env = db_env
        self.seed = seed

        webrtc_logger.info(f"Creating WebRTC demo session: {session_id}, env: {db_env.registration_id}, seed: {seed}")

        # Environment state
        self.env = None
        self.obs_buffer = []
        self.rew_buffer = []
        self.done_buffer = []
        self.info_buffer = []
        self.action_buffer = []
        self.env_initialized = False
        self.episode_done = False
        self.step_count = 0

        # WebRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.video_track: Optional[EnvironmentVideoTrack] = None
        self.data_channel: Optional[RTCDataChannel] = None

        # Control state
        self.running = False
        self.action_queue = asyncio.Queue()
        self.last_render_time = 0

        # Demo recording
        self.demo_number = self._get_next_demo_number()

        # Keyboard state for continuous control
        self.pressed_keys = set()
        self.key_mappings = self._get_key_mappings()

        # Render mode (will be determined during initialization)
        self.render_mode = None

    def _get_next_demo_number(self) -> int:
        """Find the next available demo number."""
        demo_number = 0
        while os.path.exists(os.path.join("data", "generated_demos", f"{self.session_id}_{demo_number}.npz")):
            demo_number += 1
        return demo_number

    def _get_key_mappings(self):
        """Get keyboard mappings for different environment types."""
        if "mujoco" in self.db_env.registration_id.lower() or "robosuite" in self.db_env.registration_id.lower():
            # Continuous control mappings for Mujoco/RoboSuite
            return {
                "w": [1.0, 0.0, 0.0, 0.0],  # Forward
                "s": [-1.0, 0.0, 0.0, 0.0],  # Backward
                "a": [0.0, 1.0, 0.0, 0.0],  # Left
                "d": [0.0, -1.0, 0.0, 0.0],  # Right
                "q": [0.0, 0.0, 1.0, 0.0],  # Up
                "e": [0.0, 0.0, -1.0, 0.0],  # Down
                "space": [0.0, 0.0, 0.0, 1.0],  # Gripper/Action
            }
        else:
            # Discrete action mappings for other environments
            return {
                "w": 0,  # Up/Forward
                "s": 1,  # Down/Backward
                "a": 2,  # Left
                "d": 3,  # Right
                "space": 4,  # Action/Jump
                "enter": 6,  # Done (for BabyAI)
            }

    async def initialize_environment(self):
        """Initialize the gym environment."""
        try:
            webrtc_logger.info(f"Initializing environment for session {self.session_id}")

            # Create directories
            os.makedirs(os.path.join("data", "current_demos"), exist_ok=True)
            os.makedirs(os.path.join("data", "generated_demos"), exist_ok=True)

            webrtc_logger.debug(f"Environment config: {self.exp.environment_config}")
            webrtc_logger.debug(f"Additional packages: {self.db_env.additional_gym_packages}")

            # Initialize environment with rgb_array render mode for WebRTC video streaming
            env_config = self.exp.environment_config.copy() if self.exp.environment_config else {}
            env_config["render_mode"] = "rgb_array"  # Force rgb_array mode for WebRTC

            env_wrapper = get_environment(
                self.db_env.registration_id,
                environment_config=env_config,
                n_envs=1,
                norm_env_path=None,
                additional_packages=self.db_env.additional_gym_packages,
                gym_entry_point=self.db_env.gym_entry_point,
            )

            webrtc_logger.debug(f"Environment wrapper type: {type(env_wrapper)}")

            # Get the first environment (might be wrapped)
            if hasattr(env_wrapper, "envs"):
                self.env = env_wrapper.envs[0]
                webrtc_logger.debug("Using vectorized environment")
            else:
                self.env = env_wrapper
                webrtc_logger.debug("Using single environment")

            webrtc_logger.info(f"Environment type: {type(self.env)}")

            # Reset environment
            if hasattr(self.env, "reset"):
                reset_result = self.env.reset(seed=self.seed)
                if isinstance(reset_result, tuple) and len(reset_result) >= 2:
                    obs, _ = reset_result
                else:
                    obs = reset_result
            else:
                webrtc_logger.error(f"Environment {type(self.env)} has no reset method")
                return False

            self.obs_buffer.append(obs)
            self.env_initialized = True

            # Test rendering immediately to check if it works
            webrtc_logger.debug("Testing environment rendering...")
            try:
                # Since render_mode is set at initialization, just call render() without mode
                test_render = self.env.render()

                # save a test frame as png to verify rendering
                if test_render is not None and isinstance(test_render, np.ndarray):
                    test_render = (test_render * 255).astype(np.uint8)
                    test_frame_path = os.path.join("data", "current_demos", f"{self.session_id}_test_frame.png")
                    os.makedirs(os.path.dirname(test_frame_path), exist_ok=True)
                    from PIL import Image
                    Image.fromarray(test_render).save(test_frame_path)
                else:
                    test_render = None    

                print(
                    f"Frame shape: {test_render.shape}, dtype: {test_render.dtype}, min: {test_render.min()}, max: {test_render.max()}"
                )

                if test_render is not None:
                    webrtc_logger.info(f"Test render successful: shape={test_render.shape}, dtype={test_render.dtype}")

                    self.render_mode = "human"  # We know it's rgb_array from initialization
                else:
                    webrtc_logger.warning("Environment render returned None - this is the issue!")

            except Exception as render_error:
                webrtc_logger.error(f"Test render failed: {render_error}", exc_info=True)

            webrtc_logger.info(f"Environment initialized successfully for session {self.session_id}")
            return True

        except Exception as e:
            webrtc_logger.error(f"Error initializing environment: {e}", exc_info=True)
            return False


    async def setup_webrtc(self, offer_sdp: str) -> str:
        """Setup WebRTC peer connection."""
        try:
            webrtc_logger.info(f"Setting up WebRTC for session {self.session_id}")
            
            # Log the offer to see what's requested
            webrtc_logger.debug(f"Offer requests video: {'m=video' in offer_sdp}")

            # Create peer connection
            self.pc = RTCPeerConnection()

            # Create and add video track FIRST
            self.video_track = EnvironmentVideoTrack(self.session_id)
            self.pc.addTrack(self.video_track)
            webrtc_logger.info("Added video track to peer connection")

            # Set remote description
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
            await self.pc.setRemoteDescription(offer)

            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            # VERIFY the answer includes video
            answer_sdp = answer.sdp
            if 'm=video' in answer_sdp:
                webrtc_logger.info("✓ Answer includes video track")
                # Find the video section
                video_section = answer_sdp[answer_sdp.find('m=video'):answer_sdp.find('m=', answer_sdp.find('m=video')+1)]
                webrtc_logger.debug(f"Video section in answer: {video_section[:200]}...")
            else:
                webrtc_logger.error("✗ Answer does NOT include video track!")
                
            # Also check if the track is actually ready
            webrtc_logger.info(f"Video track ID: {self.video_track.id}")
            webrtc_logger.info(f"Video track readyState: {self.video_track.readyState}")

            return answer_sdp

        except Exception as e:
            webrtc_logger.error(f"Error setting up WebRTC: {e}", exc_info=True)
            raise


    def _on_data_channel_message(self, message):
        """Handle incoming control messages."""
        try:
            data = json.loads(message)
            webrtc_logger.debug(f"Received data channel message: {data}")

            if data.get("type") == "keydown":
                key = data.get("key", "")
                self.pressed_keys.add(key)
                webrtc_logger.debug(f"Key down: {key}, pressed keys: {self.pressed_keys}")
            elif data.get("type") == "keyup":
                key = data.get("key", "")
                self.pressed_keys.discard(key)
                webrtc_logger.debug(f"Key up: {key}, pressed keys: {self.pressed_keys}")
            elif data.get("type") == "action":
                # Direct action input
                action = data.get("action")
                asyncio.create_task(self.action_queue.put(action))
                webrtc_logger.debug(f"Direct action: {action}")

        except Exception as e:
            webrtc_logger.error(f"Error processing control message: {e}", exc_info=True)

    def _get_current_action(self):
        """Convert current pressed keys to environment action."""
        if not self.pressed_keys:
            # Return neutral action based on environment type
            if "mujoco" in self.db_env.registration_id.lower():
                return [0.0] * 4  # Neutral continuous action
            else:
                return None  # No action for discrete environments

        if "mujoco" in self.db_env.registration_id.lower() or "robosuite" in self.db_env.registration_id.lower():
            # Continuous control - combine pressed keys
            action = [0.0, 0.0, 0.0, 0.0]
            for key in self.pressed_keys:
                if key in self.key_mappings:
                    key_action = self.key_mappings[key]
                    if isinstance(key_action, list) and len(key_action) >= len(action):
                        for i in range(len(action)):
                            action[i] += key_action[i]
            # Clamp action values
            action = [max(-1.0, min(1.0, a)) for a in action]
            return action
        else:
            # Discrete control - return first pressed key action
            for key in self.pressed_keys:
                if key in self.key_mappings:
                    return self.key_mappings[key]
            return None

    async def start_demo_loop(self):
        """Start the main demo loop with integrated rendering."""
        webrtc_logger.info(f"Starting demo loop for session {self.session_id}")
        self.running = True

        frame_count = 0
        render_counter = 0

        # Main control and render loop
        while self.running and not self.episode_done:
            try:
                frame_count += 1
                current_time = time.time()

                # Check for manual actions from queue
                manual_action = None
                try:
                    manual_action = await asyncio.wait_for(self.action_queue.get(), timeout=0.01)
                    if manual_action is not None:
                        webrtc_logger.debug(f"Got manual action from queue: {manual_action}")
                except asyncio.TimeoutError:
                    pass

                # Get action from keyboard or manual input
                action = manual_action if manual_action is not None else self._get_current_action()

                if action is not None:
                    webrtc_logger.debug(f"Performing action: {action} (step {self.step_count})")

                    # Perform environment step
                    if self.env is not None:
                        step_result = self.env.step(action)
                        if len(step_result) == 5:
                            obs, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        else:
                            # Handle vectorized environment
                            obs, reward, done, info = step_result
                            terminated = done
                            truncated = False

                        webrtc_logger.debug(f"Step result: reward={reward}, done={done}")

                    # Special handling for BabyAI "done" action
                    if action == 6:
                        done = True
                        webrtc_logger.debug("BabyAI done action triggered")

                    # Update buffers
                    self.obs_buffer.append(obs)
                    self.rew_buffer.append(reward)
                    self.done_buffer.append(done)
                    self.info_buffer.append(info)
                    self.action_buffer.append(action)
                    self.step_count += 1

                    if done:
                        webrtc_logger.info(f"Episode completed after {self.step_count} steps")
                        self.episode_done = True
                        await self._save_episode()

                # Render at 30 FPS (every 1/30 second)
                if current_time - self.last_render_time >= 1 / 30:
                    self.last_render_time = current_time
                    render_counter += 1

                    if self.env and self.video_track:
                        try:
                            # Render environment (render_mode was set during initialization)
                            frame = self.env.render()

                            if frame is not None:
                                if render_counter <= 5:  # Log first few frames
                                    webrtc_logger.info(
                                        f"Rendered frame {render_counter}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}"
                                    )

                                # Ensure frame is in correct format
                                if len(frame.shape) == 3 and frame.shape[2] == 3:
                                    # Convert from float to uint8 if needed
                                    if frame.max() <= 1.0:
                                        frame = (frame * 255).astype(np.uint8)
                                        if render_counter <= 5:
                                            webrtc_logger.debug("Converted frame from float to uint8")

                                    webrtc_logger.debug(f"Added frame {render_counter} to video track queue")

                                    if render_counter % 30 == 0:  # Log every second
                                        webrtc_logger.debug(
                                            f"Added frame {render_counter} to video track, shape: {frame.shape}"
                                        )
                                else:
                                    webrtc_logger.warning(f"Frame has unexpected shape: {frame.shape}")
                            else:
                                webrtc_logger.warning(f"Environment render returned None at frame {render_counter}")
                        except Exception as render_error:
                            webrtc_logger.error(f"Error rendering frame {render_counter}: {render_error}", exc_info=True)

                # Log periodic status
                if frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    webrtc_logger.debug(
                        f"Demo loop running: frame {frame_count}, step {self.step_count}, pressed keys: {self.pressed_keys}"
                    )

                # Control frame rate (30 FPS for responsive control)
                await asyncio.sleep(1 / 30)

            except Exception as e:
                webrtc_logger.error(f"Error in demo loop: {e}", exc_info=True)
                break

        webrtc_logger.info(f"Demo loop ended for session {self.session_id}")

    async def _save_episode(self):
        """Save the completed episode."""
        try:
            # Convert buffers to numpy arrays
            obs_buffer = np.array(self.obs_buffer)
            rew_buffer = np.array(self.rew_buffer)
            done_buffer = np.array(self.done_buffer)
            action_buffer = np.array(self.action_buffer)
            info_buffer = np.array(self.info_buffer)

            # Save episode
            filename = os.path.join("data", "generated_demos", f"{self.session_id}_{self.demo_number}.npz")

            with open(filename, "wb") as f:
                np.savez(
                    f,
                    obs=obs_buffer,
                    rewards=rew_buffer,
                    dones=done_buffer,
                    actions=action_buffer,
                    infos=info_buffer,
                )

            print(f"Episode saved: {filename}")

        except Exception as e:
            print(f"Error saving episode: {e}")

    async def stop(self):
        """Stop the demo session."""
        self.running = False

        if self.video_track:
            self.video_track.stop()

        if self.pc:
            await self.pc.close()

        if self.env:
            self.env.close()

async def handle_webrtc_offer(session_id: str, offer_sdp: str) -> dict:
    """Handle WebRTC offer for demo session."""
    try:
        webrtc_logger.info(f"Handling WebRTC offer for session {session_id}")

        if session_id not in _demo_sessions:
            webrtc_logger.error(f"Demo session {session_id} not found")
            return {"success": False, "error": "Demo session not found"}

        session = _demo_sessions[session_id]

        # Setup WebRTC and get answer
        answer_sdp = await session.setup_webrtc(offer_sdp)

        # Start demo loop AFTER WebRTC is setup
        task = asyncio.create_task(session.start_demo_loop())
        session._demo_task = task
        
        return {"success": True, "answer": answer_sdp}

    except Exception as e:
        webrtc_logger.error(f"Error handling WebRTC offer: {e}", exc_info=True)
        return {"success": False, "error": str(e)}



async def handle_control_message(session_id: str, message: dict) -> dict:
    """Handle control message for demo session."""
    try:
        webrtc_logger.debug(f"Handling control message for session {session_id}: {message}")

        if session_id not in _demo_sessions:
            webrtc_logger.warning(f"Demo session {session_id} not found for control message")
            return {"success": False, "error": "Demo session not found"}

        session = _demo_sessions[session_id]

        # Process control message manually since we don't have data channel
        if message.get("type") == "keydown":
            key = message.get("key", "")
            session.pressed_keys.add(key)
            webrtc_logger.debug(f"Key down: {key}, active keys: {session.pressed_keys}")
        elif message.get("type") == "keyup":
            key = message.get("key", "")
            session.pressed_keys.discard(key)
            webrtc_logger.debug(f"Key up: {key}, active keys: {session.pressed_keys}")
        elif message.get("type") == "action":
            # Direct action input
            action = message.get("action")
            await session.action_queue.put(action)
            webrtc_logger.debug(f"Direct action queued: {action}")

        return {"success": True}

    except Exception as e:
        webrtc_logger.error(f"Error handling control message: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def handle_ice_candidate(session_id: str, candidate: dict) -> dict:
    """Handle ICE candidate for demo session."""
    try:
        webrtc_logger.debug(f"Handling ICE candidate for session {session_id}: {candidate}")

        if session_id not in _demo_sessions:
            webrtc_logger.warning(f"Demo session {session_id} not found for ICE candidate")
            return {"success": False, "error": "Demo session not found"}

        session = _demo_sessions[session_id]
        
        if session.pc:
            from aiortc import RTCIceCandidate
            ice_candidate = RTCIceCandidate(
                candidate=candidate.get("candidate"),
                sdpMid=candidate.get("sdpMid"),
                sdpMLineIndex=candidate.get("sdpMLineIndex")
            )
            await session.pc.addIceCandidate(ice_candidate)
            webrtc_logger.debug(f"Added ICE candidate for session {session_id}")

        return {"success": True}

    except Exception as e:
        webrtc_logger.error(f"Error handling ICE candidate: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# Public API functions
async def create_webrtc_demo_session(session_id: str, exp: Experiment, db_env: Environment, seed: int) -> tuple[str, int]:
    """Create a new WebRTC demo session."""
    try:
        webrtc_logger.info(f"Creating WebRTC demo session: {session_id}")
        webrtc_logger.debug(f"Environment: {db_env.registration_id}, Seed: {seed}")

        # Create demo session
        demo_session = WebRTCDemoSession(session_id, exp, db_env, seed)

        # Initialize environment
        if not await demo_session.initialize_environment():
            raise Exception("Failed to initialize environment")

        # Store session
        _demo_sessions[session_id] = demo_session
        webrtc_logger.info(f"WebRTC demo session created successfully: {session_id}")
        webrtc_logger.debug(f"Active sessions: {list(_demo_sessions.keys())}")

        return session_id, demo_session.demo_number

    except Exception as e:
        webrtc_logger.error(f"Error creating WebRTC demo session: {e}", exc_info=True)
        raise


async def stop_webrtc_demo_session(session_id: str) -> bool:
    """Stop a WebRTC demo session."""
    try:
        webrtc_logger.info(f"Stopping WebRTC demo session: {session_id}")

        if session_id in _demo_sessions:
            session = _demo_sessions[session_id]
            await session.stop()
            del _demo_sessions[session_id]
            webrtc_logger.info(f"WebRTC demo session stopped successfully: {session_id}")
            webrtc_logger.debug(f"Remaining sessions: {list(_demo_sessions.keys())}")
            return True
        else:
            webrtc_logger.warning(f"WebRTC demo session {session_id} not found for stopping")
            return False

    except Exception as e:
        webrtc_logger.error(f"Error stopping WebRTC demo session: {e}", exc_info=True)
        return False

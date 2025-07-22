import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.data_models.global_models import Environment, Experiment
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("pc")

# ---------------------------------------------------------------------------
# aiortc helpers
# ---------------------------------------------------------------------------

pcs: set[RTCPeerConnection] = set()
relay = MediaRelay()
gym_sessions: Dict[str, "GymEnvironmentTrack"] = {}
data_channels: Dict[str, object] = {}


class GymEnvironmentTrack(VideoStreamTrack):
    """Video track that streams gymnasium environment renders."""

    kind = "video"

    def __init__(self, session_id: str, exp: Experiment, db_env: Environment, seed: int = 42, initial_state: Optional[dict] = None):
        super().__init__()
        self.session_id = session_id
        self.exp = exp
        self.db_env = db_env
        self.seed = seed
        self.initial_state = initial_state  # Optional state to load from
        self.counter = 0
        self.env = None
        self.obs = None
        self.episode_done = False
        self.step_count = 0
        self.initialization_done = False

        # Control state
        self.pressed_keys = set()
        self.action_queue = asyncio.Queue()
        self.key_mappings = self._get_key_mappings()
        # Debug: print(f"[GYM_TRACK] Key mappings: {list(self.key_mappings.keys())}")

        # Ensure track is properly initialized for WebRTC
        # This helps prevent the _queue=None issue
        try:
            # Force initialization of internal track state
            if hasattr(self, '_track'):
                self._track = None
        except Exception:
            pass

        # Initialize environment in background
        self._init_task = asyncio.create_task(self._initialize_environment())
        # Store task reference to avoid warning
        self._init_task.add_done_callback(lambda t: None)
        
        print(f"GymEnvironmentTrack initialized for session {session_id}")

    def _get_key_mappings(self):
        """Get keyboard mappings for different environment types."""
        # Debug: print(f"[GYM_TRACK] Environment: {self.db_env.registration_id}")
        
        if "mujoco" in self.db_env.registration_id.lower() or "metaworld" in self.db_env.registration_id.lower():
            # Metaworld/MuJoCo action space: [dx, dy, dz, gripper]
            # dx: end-effector displacement in x direction (forward/backward)
            # dy: end-effector displacement in y direction (left/right)  
            # dz: end-effector displacement in z direction (up/down)
            # gripper: gripper control (open/close)
            return {
                "w": [1.0, 0.0, 0.0, 0.0],     # Forward (positive dx)
                "s": [-1.0, 0.0, 0.0, 0.0],    # Backward (negative dx)
                "a": [0.0, 1.0, 0.0, 0.0],     # Left (positive dy)
                "d": [0.0, -1.0, 0.0, 0.0],    # Right (negative dy)
                "q": [0.0, 0.0, 0.0, -1.0],    # Gripper open (negative gripper)
                "e": [0.0, 0.0, 0.0, 1.0],     # Gripper close (positive gripper)
                "shift": [0.0, 0.0, 1.0, 0.0],    # Up (positive dz)
                "control": [0.0, 0.0, -1.0, 0.0], # Down (negative dz)
            }
        else:
            return {
                "w": 0,
                "s": 1,
                "a": 2,
                "d": 3,
                "space": 4,
                "enter": 6,
            }

    async def _initialize_environment(self):
        """Initialize gymnasium environment."""
        try:
            env_config = self.exp.environment_config.copy() if self.exp.environment_config else {}
            env_config["render_mode"] = "rgb_array"

            env_wrapper = get_environment(
                self.db_env.registration_id,
                environment_config=env_config,
                n_envs=1,
                norm_env_path=None,
                additional_packages=self.db_env.additional_gym_packages,
                gym_entry_point=self.db_env.gym_entry_point,
            )

            base_env = env_wrapper.envs[0] if hasattr(env_wrapper, "envs") else env_wrapper
            
            # Wrap with SaveResetWrapper to enable state loading
            self.env = SaveResetEnvWrapper(base_env)

            # Handle environment reset with proper seed handling
            try:
                reset_result = self.env.reset(seed=self.seed)
            except TypeError:
                # If seed parameter is not supported, try without seed
                reset_result = self.env.reset()
            
            self.obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            # Load initial state if provided
            if self.initial_state is not None:
                try:
                    logger.info(f"Loading initial state for session {self.session_id}")
                    self.obs = self.env.load_state({'state': self.initial_state, 'observation': None})
                    logger.info("Successfully loaded initial state")
                except Exception as e:
                    logger.warning(f"Failed to load initial state: {e}")
            
            self.initialization_done = True

            print(f"Environment {self.db_env.registration_id} initialized for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            self.initialization_done = True  # Mark as done even if failed

    def _resize_frame(self, frame: np.ndarray, target_width: int = 640, target_height: int = 480) -> np.ndarray:
        """Resize and pad frame to target dimensions while maintaining aspect ratio."""
        orig_height, orig_width = frame.shape[:2]

        # Calculate scaling factor to fit within target dimensions
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create output frame with target dimensions (black background)
        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Calculate padding to center the resized frame
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2

        # Place resized frame in center
        output[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized

        return output

    def _get_current_action(self):
        """Convert pressed keys to environment action."""
        if not self.pressed_keys:
            return [0.0] * 4 if ("mujoco" in self.db_env.registration_id.lower() or "metaworld" in self.db_env.registration_id.lower()) else None

        if "mujoco" in self.db_env.registration_id.lower() or "metaworld" in self.db_env.registration_id.lower():
            action = [0.0, 0.0, 0.0, 0.0]
            for key in self.pressed_keys:
                if key in self.key_mappings:
                    key_action = self.key_mappings[key]
                    for i in range(len(action)):
                        action[i] += key_action[i]
            return [max(-1.0, min(1.0, a)) for a in action]
        else:
            for key in self.pressed_keys:
                if key in self.key_mappings:
                    return self.key_mappings[key]
            return None

    async def recv(self) -> VideoFrame:
        """Generate video frame from environment."""
        pts, time_base = await self.next_timestamp()

        # Wait for initialization if not done yet
        if not self.initialization_done:
            try:
                await asyncio.wait_for(self._init_task, timeout=0.1)
            except asyncio.TimeoutError:
                pass  # Continue with loading screen

        if self.env is None or not self.initialization_done:
            # Create black frame if no environment loaded yet
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some visual indication that env is loading
            img[50:100, 50:100] = [255, 0, 0]  # Red square
        else:
            try:
                # Check for manual actions
                manual_action = None
                try:
                    manual_action = await asyncio.wait_for(self.action_queue.get(), timeout=0.001)
                except asyncio.TimeoutError:
                    pass

                # Get action from keyboard or manual input
                action = manual_action if manual_action is not None else self._get_current_action()
                
                # Only print and step if there's an actual action
                action_taken = False
                if action is not None:
                    # Check if this is a non-zero action (for continuous spaces) or valid action (for discrete)
                    is_active_action = False
                    if isinstance(action, list):
                        # For continuous action spaces (like Metaworld), check if any component is non-zero
                        is_active_action = any(abs(a) > 0.001 for a in action)
                    else:
                        # For discrete action spaces, any non-None action is considered active
                        is_active_action = True
                    
                    if is_active_action:
                        logger.debug(f"Taking action: {action}")
                        step_result = self.env.step(action)
                        if len(step_result) == 5:
                            self.obs, reward, terminated, truncated, info = step_result
                            self.episode_done = terminated or truncated
                        else:
                            self.obs, reward, self.episode_done, info = step_result

                        self.step_count += 1
                        action_taken = True

                        if self.episode_done:
                            logger.info(f"Episode completed after {self.step_count} steps")
                            # Reset environment
                            try:
                                reset_result = self.env.reset()
                            except TypeError:
                                # If reset() doesn't accept parameters, use without
                                reset_result = self.env.reset()
                            self.obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                            self.episode_done = False
                            self.step_count = 0

                # Render environment to get current state (always render for visual feedback)
                frame = self.env.render()
                if frame is not None:
                    # Only log render info when action is taken to reduce spam
                    if action_taken:
                        logger.debug(f"Environment render after action: shape={frame.shape}")

                    # very special case for metaworld (which has a bug in the rendering) - check if the env is a metaworld env
                    # and has a camera name starting with "corner"
                    # TODO: Remove this special case when the bug is fixed in metaworld/mujoco
                    try:
                        # Check for metaworld camera rotation fix
                        env_to_check = self.env
                        # Handle different environment wrapper structures
                        if hasattr(env_to_check, 'unwrapped'):
                            env_to_check = env_to_check.unwrapped
                        elif hasattr(env_to_check, 'env'):
                            env_to_check = env_to_check.env
                            if hasattr(env_to_check, 'unwrapped'):
                                env_to_check = env_to_check.unwrapped
                        
                        if hasattr(env_to_check, "camera_name") and env_to_check.camera_name.startswith("corner"):
                            # Rotate 180 degrees (2 times 90 degrees clockwise)
                            frame = np.rot90(frame, k=2)
                    except (AttributeError, TypeError):
                        # If we can't access camera_name, skip rotation
                        pass

                    # Ensure proper format
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)

                    # Resize/pad frame to target size (640x480)
                    img = self._resize_frame(frame, target_width=640, target_height=480)
                else:
                    logger.warning("Environment render returned None")
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add visual indicator for render failure
                    img[100:150, 100:150] = [0, 255, 0]  # Green square

            except Exception as e:
                logger.error(f"Error in environment step: {e}")
                img = np.zeros((480, 640, 3), dtype=np.uint8)

        self.counter += 1

        # Create video frame
        video_frame = VideoFrame.from_ndarray(img, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def handle_control_message(self, message_str: str):
        """Handle control messages from data channel."""
        try:
            data = json.loads(message_str)

            if data.get("type") == "keydown":
                key = data.get("key", "")
                self.pressed_keys.add(key)
                logger.debug(f"Key down: {key}")
                    
            elif data.get("type") == "keyup":
                key = data.get("key", "")
                self.pressed_keys.discard(key)
                logger.debug(f"Key up: {key}")
                
            elif data.get("type") == "action":
                action = data.get("action")
                task = asyncio.create_task(self.action_queue.put(action))
                # Store task reference to avoid warning
                task.add_done_callback(lambda t: None)
                logger.debug(f"Direct action: {action}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in control message: {e}")
        except Exception as e:
            logger.error(f"Error processing control message: {e}")

    def stop(self):
        """Clean up resources when track is stopped."""
        try:
            if hasattr(self, '_init_task') and not self._init_task.done():
                self._init_task.cancel()
            
            if self.env is not None:
                # Try to close the environment gracefully
                try:
                    if hasattr(self.env, 'close'):
                        self.env.close()
                except Exception as e:
                    logger.warning(f"Error closing environment: {e}")
                    
        except Exception as e:
            logger.error(f"Error during track cleanup: {e}")
        
        super().stop()

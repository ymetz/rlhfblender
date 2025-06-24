import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.data_models.global_models import Environment, Experiment

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("pc")

# ---------------------------------------------------------------------------
# aiortc helpers
# ---------------------------------------------------------------------------

pcs: set[RTCPeerConnection] = set()
relay = MediaRelay()
gym_sessions: Dict[str, "GymEnvironmentTrack"] = {}


class GymEnvironmentTrack(VideoStreamTrack):
    """Video track that streams gymnasium environment renders."""

    kind = "video"

    def __init__(self, session_id: str, exp: Experiment, db_env: Environment, seed: int = 42):
        super().__init__()
        self.session_id = session_id
        self.exp = exp
        self.db_env = db_env
        self.seed = seed
        self.counter = 0
        self.env = None
        self.obs = None
        self.episode_done = False
        self.step_count = 0

        # Control state
        self.pressed_keys = set()
        self.action_queue = asyncio.Queue()
        self.key_mappings = self._get_key_mappings()

        # Initialize environment
        asyncio.create_task(self._initialize_environment())

    def _get_key_mappings(self):
        """Get keyboard mappings for different environment types."""
        if "mujoco" in self.db_env.registration_id.lower():
            return {
                "w": [1.0, 0.0, 0.0, 0.0],
                "s": [-1.0, 0.0, 0.0, 0.0],
                "a": [0.0, 1.0, 0.0, 0.0],
                "d": [0.0, -1.0, 0.0, 0.0],
                "q": [0.0, 0.0, 1.0, 0.0],
                "e": [0.0, 0.0, -1.0, 0.0],
                "space": [0.0, 0.0, 0.0, 1.0],
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

            self.env = env_wrapper.envs[0] if hasattr(env_wrapper, "envs") else env_wrapper

            reset_result = self.env.reset(seed=self.seed)
            self.obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            logger.info(f"Environment {self.db_env.registration_id} initialized for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")

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
            return [0.0] * 4 if "mujoco" in self.db_env.registration_id.lower() else None

        if "mujoco" in self.db_env.registration_id.lower():
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

        if self.env is None:
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
                print(f"Current action: {action}")
                # Perform environment step if action available
                if action is not None:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        self.obs, reward, terminated, truncated, info = step_result
                        self.episode_done = terminated or truncated
                    else:
                        self.obs, reward, self.episode_done, info = step_result

                    self.step_count += 1

                    if self.episode_done:
                        logger.info(f"Episode completed after {self.step_count} steps")
                        # Reset environment
                        reset_result = self.env.reset(seed=self.seed + self.step_count)
                        self.obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                        self.episode_done = False
                        self.step_count = 0

                # Always render environment to get current state
                frame = self.env.render()
                if frame is not None:
                    logger.info(
                        f"Environment render successful: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}"
                    )

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
                asyncio.create_task(self.action_queue.put(action))
                logger.debug(f"Direct action: {action}")

        except Exception as e:
            logger.error(f"Error processing control message: {e}")

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper

from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.data_models.global_models import Environment, Experiment


def _sanitize_component(value: str | None) -> str:
    """Return a filesystem-friendly identifier component."""
    if not value:
        return "unknown"
    return str(value).strip().replace("/", "_").replace(" ", "-").replace(":", "-")


@dataclass
class DemoMetadata:
    """Metadata describing a recorded WebRTC demo."""

    session_id: str
    demo_number: int
    experiment_id: int | None
    experiment_name: str
    environment_id: str
    checkpoint: int | None
    projection_method: str | None
    created_at: str
    total_steps: int
    total_reward: float
    episode_rewards: list[float]
    episode_lengths: list[int]


ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("pc")

# ---------------------------------------------------------------------------
# aiortc helpers
# ---------------------------------------------------------------------------

pcs: set[RTCPeerConnection] = set()
relay = MediaRelay()
gym_sessions: dict[str, "GymEnvironmentTrack"] = {}
data_channels: dict[str, object] = {}


class GymEnvironmentTrack(VideoStreamTrack):
    """Video track that streams gymnasium environment renders."""

    kind = "video"

    def __init__(
        self,
        session_id: str,
        exp: Experiment,
        db_env: Environment,
        seed: int = 42,
        initial_state: dict | None = None,
        target_width: int = 480,
        target_height: int = 360,
        target_fps: int = 15,
    ):
        super().__init__()
        self.session_id = session_id
        self.exp = exp
        self.db_env = db_env
        self.seed = seed
        self.initial_state = initial_state  # Optional state to load from
        self.counter = 0
        self.env = None
        self.obs = None
        self.prev_obs = None
        self.episode_done = False
        self.step_count = 0
        self.initialization_done = False

        # Data logging buffers (similar to EpisodeRecorder)
        self.buffers = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "renders": [],
            "env_states": [],
            "episode_steps": [],
        }
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.demo_started = False
        self.global_step = 0
        self.demo_counter = 0

        # Control state
        self.pressed_keys = set()
        self.action_queue = asyncio.Queue()
        self.key_mappings = self._get_key_mappings()
        # Debug: print(f"[GYM_TRACK] Key mappings: {list(self.key_mappings.keys())}")

        # Ensure track is properly initialized for WebRTC
        # This helps prevent the _queue=None issue
        try:
            # Force initialization of internal track state
            if hasattr(self, "_track"):
                self._track = None
        except Exception:
            pass

        # Initialize environment in background
        self._init_task = asyncio.create_task(self._initialize_environment())
        # Store task reference to avoid warning
        self._init_task.add_done_callback(lambda t: None)

        print(f"GymEnvironmentTrack initialized for session {session_id}")

        # Streaming parameters
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = max(1, target_fps)
        self._last_sent_ts = 0.0

        # Session lifecycle tracking
        import time as _time

        self.created_at = _time.time()
        self.last_access = self.created_at
        self.stopped = False

        # Metadata for downstream processing
        self.experiment_id = getattr(exp, "id", None)
        self.experiment_name = getattr(exp, "exp_name", "")
        self.environment_id = getattr(db_env, "registration_id", "")
        self.current_checkpoint: int | None = None
        self.projection_method: str | None = None
        self.projection_props: dict | None = None
        self.last_saved_path: Path | None = None
        self.last_saved_metadata_path: Path | None = None

    def touch(self):
        """Update last access time for session TTL handling."""
        import time as _time

        self.last_access = _time.time()

    def _get_key_mappings(self):
        """Get keyboard mappings for different environment types."""
        # Debug: print(f"[GYM_TRACK] Environment: {self.db_env.registration_id}")

        if "mujoco" in self.db_env.registration_id.lower() or "metaworld" in self.db_env.registration_id.lower():
            # Metaworld/MuJoCo action space: [dx, dy, dz, gripper]
            # dx: end-effector displacement in x direction (forward/backward)
            # dy: end-effector displacement in y direction (left/right)
            # dz: end-effector displacement in z direction (up/down)
            # gripper: gripper control (open/close)

            # we flip dx and dy to match the camera angle (x is forward, y is left)
            return {
                "w": [0.0, 1.0, 0.0, 0.0],  # Forward (positive dx)
                "s": [0.0, -1.0, 0.0, 0.0],  # Backward (negative dx)
                "a": [-1.0, 0.0, 0.0, 0.0],  # Left (positive dy)
                "d": [1.0, 0.0, 0.0, 0.0],  # Right (negative dy)
                "q": [0.0, 0.0, 0.0, -1.0],  # Gripper open (negative gripper)
                "e": [0.0, 0.0, 0.0, 1.0],  # Gripper close (positive gripper)
                "shift": [0.0, 0.0, 1.0, 0.0],  # Up (positive dz)
                "control": [0.0, 0.0, -1.0, 0.0],  # Down (negative dz)
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
            # check if separate demo_gen_environment_config exists
            if self.exp.demo_gen_environment_config:
                env_config = self.exp.demo_gen_environment_config.copy()
            else:
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
                    self.obs = self.env.load_state({"state": self.initial_state, "observation": None})
                    logger.info("Successfully loaded initial state")
                except Exception as e:
                    logger.warning(f"Failed to load initial state: {e}")

            # Log initial observation (required for episode_recorder format)
            self._log_initial_observation()

            self.initialization_done = True

            print(f"Environment {self.db_env.registration_id} initialized for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            self.initialization_done = True  # Mark as done even if failed

    def _resize_frame(
        self, frame: np.ndarray, target_width: int | None = None, target_height: int | None = None
    ) -> np.ndarray:
        """Resize and pad frame to target dimensions while maintaining aspect ratio."""
        if target_width is None:
            target_width = self.target_width
        if target_height is None:
            target_height = self.target_height
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

    def _log_initial_observation(self):
        """Log the initial observation to start the episode recording."""
        if self.obs is not None:
            self.prev_obs = np.array(self.obs)

        # Enable logging even if the initial reset did not yield an immediate observation
        # so the first user action is never discarded.
        self.demo_started = True
        logger.info(f"Started data logging for demo session {self.session_id}")

    def _log_step(self, prev_obs, action, reward, done, info, render_frame, step_index: int):
        """Log step data to buffers in episode_recorder format."""
        if not self.demo_started:
            return

        if prev_obs is None and self.prev_obs is not None:
            prev_obs = self.prev_obs

        if prev_obs is not None:
            self.buffers["obs"].append(np.array(prev_obs))
            self.buffers["episode_steps"].append(step_index)

        # Log step data
        self.buffers["actions"].append(np.array(action) if action is not None else np.array([]))
        self.buffers["rewards"].append(float(reward) if reward is not None else 0.0)
        self.buffers["dones"].append(bool(done))
        self.buffers["infos"].append(info if info is not None else {})

        # Log render frame if available
        if render_frame is not None:
            self.buffers["renders"].append(render_frame.copy())
        else:
            # Create placeholder frame
            self.buffers["renders"].append(np.zeros((480, 640, 3), dtype=np.uint8))

        # Log environment state if supported
        if hasattr(self.env, "save_state"):
            try:
                env_state = self.env.save_state()
                self.buffers["env_states"].append(env_state)
            except Exception as e:
                logger.warning(f"Failed to save environment state: {e}")
                self.buffers["env_states"].append(None)
        else:
            self.buffers["env_states"].append(None)

        # Update episode tracking
        self.current_episode_reward += float(reward) if reward is not None else 0.0
        self.current_episode_length += 1

        # Remember last observation so we can fall back if the next step cannot
        # capture it synchronously (e.g., due to async scheduling).
        if self.obs is not None:
            self.prev_obs = np.array(self.obs)

        # If episode is done, record episode stats
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            logger.info(f"Episode completed: reward={self.current_episode_reward}, length={self.current_episode_length}")

            # Reset for potential next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

            # Log the reset observation for next episode
            if self.obs is not None:
                # Ensure the next step will record starting observation
                self.prev_obs = np.array(self.obs)

    def save_demo_data(self, demo_number: int = 0) -> Path | None:
        """Save the recorded demo data in episode_recorder format.

        Returns the path to the saved `.npz` file on success, otherwise ``None``.
        """
        if not self.demo_started or len(self.buffers["actions"]) == 0:
            logger.warning(f"No demo data to save for session {self.session_id}")
            return None

        try:
            total_steps = len(self.buffers["actions"])
            total_reward = float(np.sum(self.buffers["rewards"])) if self.buffers["rewards"] else 0.0

            # Convert buffers to numpy arrays (similar to EpisodeRecorder.finalize_buffers)
            final_buffers = {}
            obs_buffer = self.buffers["obs"]
            episode_steps_buffer = self.buffers["episode_steps"]

            if len(obs_buffer) and len(obs_buffer) != total_steps:
                # Align observations with recorded actions by truncating/ padding if necessary
                if len(obs_buffer) > total_steps:
                    obs_buffer = obs_buffer[:total_steps]
                else:
                    obs_buffer = obs_buffer + [obs_buffer[-1]] * (total_steps - len(obs_buffer))

            if len(episode_steps_buffer) != total_steps:
                if len(episode_steps_buffer) > total_steps:
                    episode_steps_buffer = episode_steps_buffer[:total_steps]
                else:
                    start_idx = episode_steps_buffer[-1] + 1 if episode_steps_buffer else 0
                    episode_steps_buffer = episode_steps_buffer + list(
                        range(start_idx, start_idx + (total_steps - len(episode_steps_buffer)))
                    )

            # Handle observations
            if len(obs_buffer) > 0:
                if len(np.array(obs_buffer[0]).shape) == 2:
                    final_buffers["obs"] = np.expand_dims(np.array(obs_buffer), axis=-1)
                else:
                    final_buffers["obs"] = np.array(obs_buffer)
            else:
                final_buffers["obs"] = np.array([])

            # Convert other buffers
            final_buffers["actions"] = np.array(self.buffers["actions"])
            final_buffers["rewards"] = np.array(self.buffers["rewards"])
            final_buffers["dones"] = np.array(self.buffers["dones"])
            final_buffers["infos"] = np.array(self.buffers["infos"], dtype=object)
            final_buffers["renders"] = np.array(self.buffers["renders"]) if len(self.buffers["renders"]) > 0 else np.zeros(1)
            final_buffers["env_states"] = np.array(self.buffers["env_states"], dtype=object)

            # Include episode summary data
            episode_rewards = (
                np.array(self.episode_rewards) if self.episode_rewards else np.array([self.current_episode_reward])
            )
            episode_lengths = (
                np.array(self.episode_lengths) if self.episode_lengths else np.array([self.current_episode_length])
            )

            # Construct save path with metadata-rich naming
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            env_component = _sanitize_component(self.environment_id)
            exp_component = f"exp-{self.experiment_id}" if self.experiment_id is not None else "exp-unknown"
            checkpoint_component = (
                f"checkpoint-{self.current_checkpoint}" if self.current_checkpoint is not None else "checkpoint-unset"
            )
            base_dir = Path("data") / "generated_demos" / env_component / exp_component / checkpoint_component
            base_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{_sanitize_component(self.session_id)}_demo-{demo_number:03d}_{timestamp}.npz"
            save_path = base_dir / filename

            # Save in episode_recorder format
            with open(save_path, "wb") as f:
                np.savez(
                    f,
                    obs=final_buffers["obs"],
                    actions=final_buffers["actions"],
                    rewards=final_buffers["rewards"],
                    dones=final_buffers["dones"],
                    infos=final_buffers["infos"],
                    renders=final_buffers["renders"],
                    env_states=final_buffers["env_states"],
                    episode_steps=np.array(episode_steps_buffer),
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    additional_metrics={},  # Empty dict for compatibility
                )

            metadata = DemoMetadata(
                session_id=self.session_id,
                demo_number=demo_number,
                experiment_id=self.experiment_id,
                experiment_name=self.experiment_name,
                environment_id=self.environment_id,
                checkpoint=self.current_checkpoint,
                projection_method=self.projection_method,
                created_at=timestamp,
                total_steps=int(total_steps),
                total_reward=float(total_reward),
                episode_rewards=episode_rewards.tolist(),
                episode_lengths=episode_lengths.tolist(),
            )

            metadata_path = save_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(asdict(metadata), meta_file, indent=2)

            self.last_saved_path = save_path
            self.last_saved_metadata_path = metadata_path
            self.demo_counter = max(self.demo_counter, demo_number + 1)

            logger.info(f"Saved demo data to {save_path}")
            logger.info(f"Demo stats: {len(final_buffers['actions'])} steps, {len(episode_rewards)} episodes")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save demo data: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _get_current_action(self):
        """Convert pressed keys to environment action."""
        if not self.pressed_keys:
            return (
                [0.0] * 4
                if ("mujoco" in self.db_env.registration_id.lower() or "metaworld" in self.db_env.registration_id.lower())
                else None
            )

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

        # Throttle frame production to target_fps to reduce CPU/bandwidth
        loop = asyncio.get_event_loop()
        now = loop.time()
        min_interval = 1.0 / self.target_fps
        sleep_needed = (self._last_sent_ts + min_interval) - now
        if sleep_needed > 0:
            await asyncio.sleep(sleep_needed)
        self._last_sent_ts = loop.time()

        if self.env is None or not self.initialization_done:
            # Create black frame if no environment loaded yet
            img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
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
                        prev_obs = np.array(self.obs) if self.obs is not None else None
                        step_result = self.env.step(action)
                        if len(step_result) == 5:
                            self.obs, reward, terminated, truncated, info = step_result
                            self.episode_done = terminated or truncated
                        else:
                            self.obs, reward, self.episode_done, info = step_result

                        self.step_count += 1
                        action_taken = True

                        # Get render frame for logging (before any transformations)
                        log_frame = self.env.render()

                        # Log the step data
                        self._log_step(prev_obs, action, reward, self.episode_done, info, log_frame, self.global_step)
                        self.global_step += 1

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
                        if hasattr(env_to_check, "unwrapped"):
                            env_to_check = env_to_check.unwrapped
                        elif hasattr(env_to_check, "env"):
                            env_to_check = env_to_check.env
                            if hasattr(env_to_check, "unwrapped"):
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

                    # Resize/pad frame to target size
                    img = self._resize_frame(frame)
                else:
                    logger.warning("Environment render returned None")
                    img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                    # Add visual indicator for render failure
                    img[100:150, 100:150] = [0, 255, 0]  # Green square

            except Exception as e:
                logger.error(f"Error in environment step: {e}")
                img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

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
            if hasattr(self, "_init_task") and not self._init_task.done():
                self._init_task.cancel()

            if self.env is not None:
                # Try to close the environment gracefully
                try:
                    if hasattr(self.env, "close"):
                        self.env.close()
                except Exception as e:
                    logger.warning(f"Error closing environment: {e}")

        except Exception as e:
            logger.error(f"Error during track cleanup: {e}")

        self.stopped = True
        super().stop()


def create_render_from_state(exp: Experiment, db_env: Environment, state: dict, seed: int = 42) -> np.ndarray:
    """
    Create a single render frame from a given state without setting up WebRTC.

    Args:
        exp: Experiment object with environment configuration
        db_env: Database environment object
        state: Environment state to load
        seed: Random seed for environment

    Returns:
        numpy array of rendered frame
    """
    # Create environment
    env_wrapper = get_environment(
        env_name=exp.env_id,
        environment_config=exp.environment_config,
        n_envs=1,
        additional_packages=db_env.additional_gym_packages,
        gym_entry_point=db_env.gym_entry_point,
    )

    # Get the base environment and wrap with SaveResetEnvWrapper
    base_env = env_wrapper.envs[0] if hasattr(env_wrapper, "envs") else env_wrapper
    env = SaveResetEnvWrapper(base_env)

    try:
        # Reset environment first
        env.reset(seed=seed)

        # Load the state
        env.load_state({"state": state, "observation": None})

        # Render the environment
        render_frame = env.render()

        env_to_check = env
        # Handle different environment wrapper structures
        if hasattr(env_to_check, "unwrapped"):
            env_to_check = env_to_check.unwrapped
        elif hasattr(env_to_check, "env"):
            env_to_check = env_to_check.env
            if hasattr(env_to_check, "unwrapped"):
                env_to_check = env_to_check.unwrapped

        if hasattr(env_to_check, "camera_name") and env_to_check.camera_name.startswith("corner"):
            # Rotate 180 degrees (2 times 90 degrees clockwise)
            render_frame = np.rot90(render_frame, k=2)

        return render_frame

    finally:
        # Clean up
        try:
            env.close()
        except:
            pass

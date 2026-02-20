from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import Future
from io import BytesIO
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


class _DashRuntime:
    """Runs Playwright async API on a dedicated event loop thread."""

    def __init__(
        self,
        *,
        url: str,
        headless: bool,
        rl_config: dict[str, Any] | None = None,
        viewport_width: int = 768,
        viewport_height: int = 768,
    ) -> None:
        self.url = url
        self.headless = headless
        self.rl_config = rl_config
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._startup_error: Exception | None = None

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self, timeout: float = 45.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="dash-playwright-runtime", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("Timed out while starting Dash Playwright runtime")
        if self._startup_error is not None:
            raise RuntimeError(f"Failed to start Dash Playwright runtime: {self._startup_error}") from self._startup_error

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ainit())
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _ainit(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )

        # Prevent Dash welcome modal in fresh sessions
        await self._context.add_init_script("() => { window.localStorage.setItem('dash_WelcomeModal', 'hide'); }")

        self._page = await self._context.new_page()
        await self._page.goto(self.url, wait_until="domcontentloaded")

        # Optional hardening in case modal was already toggled by timing/race
        await self._page.evaluate("""
            () => {
            window.localStorage.setItem('dash_WelcomeModal', 'hide');
            const modal = document.getElementById('welcome-modal');
            if (modal) modal.classList.remove('is-active');
            }
        """)

        await self._page.wait_for_function(
            "window.simulator && typeof window.simulator.envReset === 'function' && typeof window.simulator.envStep === 'function'"
        )

        if self.rl_config:
            await self._page.evaluate("(cfg) => window.simulator.setRLConfig(cfg)", self.rl_config)

    def _submit(self, coro: Any, timeout: float = 30.0) -> Any:
        if self._loop is None:
            raise RuntimeError("Dash runtime loop is not running")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def env_reset(self, options: dict[str, Any]) -> dict[str, Any]:
        return self._submit(self._page.evaluate("(opts) => window.simulator.envReset(opts)", options))

    def env_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._submit(self._page.evaluate("(a) => window.simulator.envStep(a)", payload))

    def screenshot_png(self) -> bytes:
        return self._submit(self._page.screenshot(type="png"))

    def close(self, timeout: float = 20.0) -> None:
        if self._loop is None:
            return

        async def _aclose() -> None:
            if self._context is not None:
                await self._context.close()
                self._context = None
            if self._browser is not None:
                await self._browser.close()
                self._browser = None
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

        try:
            self._submit(_aclose(), timeout=timeout)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=timeout)
            self._thread = None
            self._loop = None


class DashDrivingGymEnv(gym.Env):
    """
    Gymnasium wrapper around a Dash iframe/player runtime exposed through
    `window.simulator.envReset` and `window.simulator.envStep`.
    """

    metadata = {"render_modes": ["human", "none", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        url: str | None = None,
        render_mode: str = "none",
        headless: bool | None = None,
        default_reset_options: dict[str, Any] | None = None,
        rl_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}")

        self.url = url or os.environ.get("DASH_PLAYER_URL", "http://localhost:5173")
        self.render_mode = render_mode
        # Keep browser headless for non-human modes by default.
        self.headless = (render_mode != "human") if headless is None else headless
        self.default_reset_options = default_reset_options or {"startMode": "manual", "clearRecording": True}
        self._rl_config = rl_config

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._runtime: _DashRuntime | None = None

        # [steer, gas, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # [speed, steering, station, latitude, heading_error, curvature, nearest_obstacle_distance, offroad, collision, sim_time]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, -np.inf, -np.inf, -np.pi, -np.inf, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf, np.pi, np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self._last_obs = np.zeros((10,), dtype=np.float32)

    @property
    def page(self):
        if self._page is None:
            raise RuntimeError("Dash driving runtime is not initialized")
        return self._page

    def _ensure_runtime(self) -> None:
        if self._runtime is not None:
            return
        self._runtime = _DashRuntime(url=self.url, headless=self.headless, rl_config=self._rl_config)
        self._runtime.start()

    @staticmethod
    def _obs_to_array(obs: dict[str, Any]) -> np.ndarray:
        nearest = obs.get("nearestObstacleDistance")
        if nearest is None:
            nearest = 1e6

        return np.array(
            [
                float(obs.get("speed", 0.0)),
                float(obs.get("steering", 0.0)),
                float(obs.get("station", 0.0)),
                float(obs.get("latitude", 0.0)),
                float(obs.get("headingError", 0.0)),
                float(obs.get("curvature", 0.0)),
                float(nearest),
                1.0 if bool(obs.get("offroad", False)) else 0.0,
                1.0 if bool(obs.get("collision", False)) else 0.0,
                float(obs.get("time", 0.0)),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._ensure_runtime()

        reset_options = dict(self.default_reset_options)
        if options:
            reset_options.update(options)

        result = self._runtime.env_reset(reset_options)
        self._last_obs = self._obs_to_array(result.get("observation", {}))
        info = result.get("info", {}) if isinstance(result, dict) else {}
        return self._last_obs.copy(), info

    def step(self, action):
        self._ensure_runtime()
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        payload = {"steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2])}
        result = self._runtime.env_step(payload)

        self._last_obs = self._obs_to_array(result.get("observation", {}))
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", {})
        return self._last_obs.copy(), reward, terminated, truncated, info

    def save_state(self):
        # Simulator-level serialization is optional; use observation fallback for compatibility.
        return {"observation": self._last_obs.copy()}

    def get_state(self):
        # Used by SaveResetEnvWrapper to avoid deepcopy of environment internals.
        return self.save_state()

    def load_state(self, state):
        # State restoration is not yet available in the iframe integration.
        obs = state.get("observation") if isinstance(state, dict) else None
        if obs is not None:
            self._last_obs = np.asarray(obs, dtype=np.float32)
        return self._last_obs.copy()

    def set_state(self, state):
        # Used by SaveResetEnvWrapper.
        self.load_state(state)

    def __getstate__(self):
        """
        Make env deepcopy/pickle safe by excluding non-serializable runtime handles.
        """
        state = self.__dict__.copy()
        state["_runtime"] = None
        state["_playwright"] = None
        state["_browser"] = None
        state["_context"] = None
        state["_page"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def render(self):
        if self.render_mode == "none":
            return None

        self._ensure_runtime()
        try:
            png_bytes = self._runtime.screenshot_png()
            img = Image.open(BytesIO(png_bytes)).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            # Keep downstream recorders robust if screenshots fail transiently.
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._page = None

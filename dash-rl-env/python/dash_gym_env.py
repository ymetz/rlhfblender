from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Playwright, sync_playwright


class DashGymEnv(gym.Env):
    """Gymnasium wrapper for Dash via Playwright."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        url: str = "http://127.0.0.1:5173",
        render_mode: str = "none",
        headless: bool = True,
        start_dev_server: bool = False,
        dev_server_cwd: str | None = None,
        dev_server_cmd: list[str] | None = None,
        rl_config: dict[str, Any] | None = None,
        default_reset_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}")

        self.url = url
        self.render_mode = render_mode
        self.headless = headless if render_mode == "none" else False
        self.start_dev_server = start_dev_server
        self.dev_server_cwd = Path(dev_server_cwd or Path(__file__).resolve().parents[1])
        self.dev_server_cmd = dev_server_cmd or [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            "5173",
        ]
        self.default_reset_options = default_reset_options or {
            "scenarioName": "rough_road",
            "startMode": "manual",
            "clearRecording": True,
        }

        self._playwright: Playwright | None = None
        self._browser = None
        self._context = None
        self._page = None
        self._dev_server_proc: subprocess.Popen[str] | None = None

        # steer, gas, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # [speed, steering, station, latitude, heading_error, curvature,
        #  nearest_obstacle_distance, offroad, collision, time]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, -np.inf, -np.inf, -np.pi, -np.inf, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf, np.pi, np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self._start_runtime()

        if rl_config:
            self.page.evaluate("(cfg) => window.simulator.setRLConfig(cfg)", rl_config)

    @property
    def page(self):
        if self._page is None:
            raise RuntimeError("Environment is closed")
        return self._page

    def _start_runtime(self) -> None:
        if self.start_dev_server:
            self._dev_server_proc = subprocess.Popen(
                self.dev_server_cmd,
                cwd=self.dev_server_cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._wait_for_http(self.url, timeout_s=30)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded")
        self._page.wait_for_function(
            "window.simulator && typeof window.simulator.envReset === 'function' && typeof window.simulator.envStep === 'function'"
        )

    @staticmethod
    def _wait_for_http(url: str, timeout_s: int) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except Exception:
                time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for {url}")

    @staticmethod
    def _obs_to_array(obs: dict[str, Any]) -> np.ndarray:
        nearest = obs.get("nearestObstacleDistance")
        if nearest is None:
            nearest = 1e6

        vec = np.array(
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
        return vec

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        reset_options = dict(self.default_reset_options)
        if options:
            reset_options.update(options)

        result = self.page.evaluate("(opts) => window.simulator.envReset(opts)", reset_options)
        obs = self._obs_to_array(result["observation"])
        info = result.get("info", {})
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        payload = {"steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2])}
        result = self.page.evaluate("(a) => window.simulator.envStep(a)", payload)

        obs = self._obs_to_array(result["observation"])
        reward = float(result["reward"])
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        info = result.get("info", {})
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def start_human_recording(self, sample_hz: int = 10, require_manual_mode: bool = True) -> dict[str, Any]:
        return self.page.evaluate(
            "(opts) => window.simulator.startHumanTeleopRecording(opts)",
            {"sampleHz": int(sample_hz), "clear": True, "requireManualMode": bool(require_manual_mode)},
        )

    def stop_human_recording(self) -> dict[str, Any]:
        return self.page.evaluate("() => window.simulator.stopHumanTeleopRecording()")

    def save_human_recording(self, path: str) -> str:
        data = self.page.evaluate("() => window.simulator.getHumanTeleopRecording()")
        output_path = Path(path)
        output_path.write_text(json.dumps(data), encoding="utf-8")
        return str(output_path)

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        if self._dev_server_proc is not None:
            self._dev_server_proc.terminate()
            try:
                self._dev_server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._dev_server_proc.kill()
            self._dev_server_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

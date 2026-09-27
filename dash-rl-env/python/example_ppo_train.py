from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from dash_gym_env import DashGymEnv


def main() -> None:
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppo_dash"

    env = Monitor(
        DashGymEnv(
            start_dev_server=True,
            render_mode="none",
            headless=True,
            default_reset_options={"scenarioName": "rough_road", "startMode": "manual", "clearRecording": True},
            rl_config={
                "dt": 1 / 60,
                "actionRepeat": 6,  # 10 Hz policy control at 60 Hz physics
                "maxSteps": 1500,
                "stepPenalty": 0.01,
                "collisionPenalty": 12.0,
            },
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
    )
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(str(model_path))

    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(f"Saved PPO model to: {model_path}.zip")


if __name__ == "__main__":
    main()

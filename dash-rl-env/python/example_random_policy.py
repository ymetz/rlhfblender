import numpy as np

from dash_gym_env import DashGymEnv


def main():
    with DashGymEnv(start_dev_server=True, render_mode="none", headless=True) as env:
        obs, info = env.reset(options={"startMode": "manual"})
        print("reset:", obs.shape, info)

        total_reward = 0.0
        for t in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"episode finished at step={t}, reward={total_reward:.3f}")
                break

        # Optional human teleop recording helpers
        # env.start_human_recording(sample_hz=10)
        # ... drive manually in a headed browser ...
        # data = env.stop_human_recording()
        # print("samples:", len(data["frames"]))


if __name__ == "__main__":
    np.random.seed(0)
    main()


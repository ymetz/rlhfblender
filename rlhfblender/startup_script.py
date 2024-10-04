"""
Ensure that the data for the demo is available by running benchmarks with pre-trained models.
This script runs benchmarks with the provided models and then creates video/thumbnail/reward data.
"""

import asyncio
import os
import traceback

from rlhfblender.utils.data_generation import generate_data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Run benchmarks for Atari Breakout
    benchmark_dicts = [
        {
            "env": "ALE/Breakout-v5",
            "benchmark_type": "trained",
            "exp": "Atari Breakout",
            "checkpoint_step": i * 1000000,
            "n_episodes": 1,
            "path": os.path.join("rlhfblender_demo_models/Atari Breakout"),
            "env_kwargs": {
                "env_wrapper": "stable_baselines3.common.atari_wrappers.AtariWrapper",
                "frame_stack": 4,
            },
            "framework": "StableBaselines3",
            "project": "RLHF-Blender",
        }
        for i in range(2, 12, 2)
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Atari Breakout benchmarks:", e)

    # Run benchmarks for BabyAI
    benchmark_dicts = [
        {
            "env": "BabyAI-MiniBossLevel-v0",
            "benchmark_type": "trained",
            "exp": "BabyAI",
            "checkpoint_step": 10000000,
            "n_episodes": 20,
            "path": os.path.join("rlhfblender_demo_models/BabyAI"),
            "framework": "BabyAI",
            "project": "RLHF-Blender",
        }
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running BabyAI benchmarks:", e)

    # Run benchmarks for Highway-Env
    benchmark_dicts = [
        {
            "env": "roundabout-v0",
            "benchmark_type": "trained",
            "exp": "Highway_env",
            "checkpoint_step": (i + 1) * 4000,
            "n_episodes": 10,
            "path": os.path.join("rlhfblender_demo_models/Highway_env"),
            "framework": "StableBaselines3",
            "project": "RLHF-Blender",
        }
        for i in range(3)
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Highway-Env benchmarks:", e)

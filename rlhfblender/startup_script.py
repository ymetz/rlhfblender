"""
Ensure that the data for the demo is available by running benchmarks with pre-trained models.
This script runs benchmarks with the provided models and then creates video/thumbnail/reward data.
"""

import asyncio
import glob
import os
import traceback

from rlhfblender.utils.data_generation import generate_data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # look for env variable FORCE_RECREATE_DATA to force data recreation
    force_recreate_data = os.getenv("RLHFLBENDER_FORCE_RECREATE_DATA", False)

    # Run benchmarks for Atari Breakout
    benchmark_dicts = [
        {
            "env": "ALE/Breakout-v5",
            "benchmark_type": "trained",
            "exp": "ALE/Breakout-v5_trained",
            "checkpoint_step": i * 1000000,
            "n_episodes": 5,
            "path": os.path.join("rlhfblender_demo_models/ALE-Breakout-v5"),
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
        # if data does not exist, any directory of signature Breakout-v5_** is considered as a valid directory
        if not glob.glob(os.path.join("data", "episodes", "ALE-Breakout-v5", "ALE-Breakout-v5_*")) or force_recreate_data:
            asyncio.run(generate_data(benchmark_dicts))
        else:
            print("Data already exists for Atari Breakout")
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Atari Breakout benchmarks:", e)

    # Run benchmarks for BabyAI
    benchmark_dicts = [
        {
            "env": "BabyAI-MiniBossLevel-v0",
            "benchmark_type": "random", # need to re-train with minigrid
            "exp": "BabyAI-MiniBossLevel-v0_random",
            "checkpoint_step": 10000000,
            "n_episodes": 20,
            "path": os.path.join("rlhfblender_demo_models/BabyAI"),
            "framework": "BabyAI",
            "project": "RLHF-Blender",
        }
    ]

    try:
        if (
            not glob.glob(os.path.join("data", "episodes", "BabyAI-MiniBossLevel-v0", "BabyAI-MiniBossLevel-v0_*"))
            or force_recreate_data
        ):
            asyncio.run(generate_data(benchmark_dicts))
        else:
            print("Data already exists for BabyAI")
    except Exception as e:
        print(traceback.format_exc())
        print("Error running BabyAI benchmarks:", e)

    # Run benchmarks for Highway-Env
    benchmark_dicts = [
        {
            "env": "roundabout-v0",
            "benchmark_type": "trained",
            "exp": "roundabout-v0_trained",
            "checkpoint_step": (i + 1) * 100000,
            "n_episodes": 10,
            "path": os.path.join("rlhfblender_demo_models/roundabout-v0"),
            "framework": "StableBaselines3",
            "project": "RLHF-Blender",
        }
        for i in range(5)
    ]

    try:
        if not glob.glob(os.path.join("data", "episodes", "roundabout-v0", "roundabout-v0_*")) or force_recreate_data:
            asyncio.run(generate_data(benchmark_dicts))
        else:
            print("Data already exists for Highway-Env")
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Highway-Env benchmarks:", e)

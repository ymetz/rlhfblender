"""
Make sure that the data for the demo is available, run at application startup.
Expects pre-trained models in the experimentation directory. First runs benchmarks with the provided
models, then creates video/thumbnail/reward data etc.
"""

import asyncio
import os
import traceback

from rlhfblender.data_collection import framework_selector as framework_selector
from rlhfblender.register import register_env, register_experiment
from rlhfblender.utils.data_generation import generate_data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Register environments
    asyncio.run(register_env("ALE/Breakout-v5"))
    asyncio.run(register_env("BabyAI-MiniBossLevel-v0"))
    asyncio.run(register_env("roundabout-v0"))

    # Register experiments
    asyncio.run(
        register_experiment(
            "Atari Breakout",
            env_id="ALE/Breakout-v5",
            env_kwargs={"env_wrapper": "stable_baselines3.common.atari_wrappers.AtariWrapper", "frame_stack": 4},
        )
    )
    asyncio.run(register_experiment("BabyAI", env_id="BabyAI-MiniBossLevel-v0", framework="BabyAI"))
    asyncio.run(register_experiment("Highway_env", env_id="roundabout-v0"))

    # Run benchmarks for Atari-Breakout
    benchmark_dicts = [
        {
            "env_id": "ALE/Breakout-v5",
            "benchmark_type": "trained",
            "benchmark_id": "Atari Breakout",
            "checkpoint_step": i * 1000000,
            "n_episodes": 1,
            "path": os.path.join("rlhfblender_demo_models/Atari Breakout"),
        }
        for i in range(2, 12, 2)
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Atari Breakout benchmarks: ", e)

    # Run benchmarks for BabyAI
    benchmark_dicts = [
        {
            "env_id": "BabyAI-MiniBossLevel-v0",
            "benchmark_type": "trained",
            "benchmark_id": "BabyAI",
            "checkpoint_step": 10000000,
            "n_episodes": 20,
            "path": os.path.join("rlhfblender_demo_models/BabyAI"),
        }
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running BabyAI benchmarks: ", e)

    # Run benchmarks for Highway-Env
    benchmark_dicts = [
        {
            "env_id": "roundabout-v0",
            "benchmark_type": "trained",
            "benchmark_id": "Highway_env",
            "checkpoint_step": (i + 1) * 4000,
            "n_episodes": 10,
            "path": os.path.join("rlhfblender_demo_models/Highway_env"),
        }
        for i in range(3)
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(traceback.format_exc())
        print("Error running Highway-Env benchmarks: ", e)

import asyncio
import os

import safety_gym
from pydantic import BaseModel
from routes.data import run_benchmark


class BenchmarkRequestModel(BaseModel):
    """
    A request model for a single benchmark run
    """

    env_id: str = ""
    gym_registration_id: str = ""
    path: str = ""
    benchmark_type: str = "random"
    benchmark_id: int = -1
    checkpoint_step: int = -1
    n_episodes: int = 1
    force_overwrite: bool = False
    render: bool = True
    deterministic: bool = False
    reset_state: bool = False
    split_by_episode: bool = False


async def run_benchmark_async(benchmark_dicts):
    request = [BenchmarkRequestModel(**item) for item in benchmark_dicts]
    result = await run_benchmark(request)
    print("RESULT:", result)


if __name__ == "__main__":
    benchmark_dicts = [
        {
            "env_id": -1,  # For which environment to run the benchmark
            "gym_registration_id": "Safexp-PointButton2-v0",  # no clue what this is
            "benchmark_type": "trained",  # Which type of benchmark to run (trained model or random agent)
            "benchmark_id": 6,  # The id of the benchmark (can just be an integer here)
            "checkpoint_step": (i + 1)
            * 1000000,  # Which checkpoint to use for the trained model (see exeriments folder for all checkpoints)
            "n_episodes": 5,  # How many episodes to we collect for each checkpoint?
            "path": os.path.join(
                f"experiments/Safety_gym"
            ),  # Where to find the trained model
            "render": True,
            "force_overwrite": True,
            "split_by_episode": True,  # If we want to split the data by episode (True) or not (False), i think for its good to turn it on
            "record_episode_videos": True,  # If we want to record videos of the episodes (True) or not (False)
        }
        for i in range(10)
    ]

    asyncio.run(run_benchmark_async(benchmark_dicts))

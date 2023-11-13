import asyncio
import os

from pydantic import BaseModel

from rlhfblender.routes.data import run_benchmark


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

"""
Pre-generated data for the RLHFBlender project (e.g. record videos, reward logs, etc.)
These can the be loaded in the user interface for studies
"""
import argparse
import asyncio
import sys

from rlhfblender.data_collection import framework_selector as framework_selector
from rlhfblender.utils.data_generation import generate_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for RLHFBlender")
    parser.add_argument("--exp", type=str, help="The experiment name.")
    parser.add_argument("--num-episodes", type=int, help="The number of episodes to run.", default=10)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", action="store_true")
    group.add_argument("--model-path", type=str, default="")

    parser.add_argument("--checkpoints", type=str, nargs="+", default="-1", help="The checkpoint steps to use.")

    parser.add_argument("--env", type=str, help="(Optional) Overwriting the env. associated to the experiment.", default=None)

    args = parser.parse_args()

    # make sure that env and exp are set
    if args.exp is None:
        print("Please specify an environment and an experiment to generate data for.")
        sys.exit(1)

    if args.random:
        args.model_path = ""
        checkpoints = ["-1"]
    else:
        checkpoints = args.checkpoints

    benchmark_dicts = [
        {
            "env_id": args.env,
            "benchmark_type": "trained" if not args.random else "random",
            "benchmark_id": args.exp,
            "checkpoint_step": checkpoint,
            "n_episodes": args.num_episodes,
            "path": args.model_path,
        }
        for checkpoint in checkpoints
    ]

    asyncio.run(generate_data(benchmark_dicts))

"""
Pre-generated data for the RLHFBlender project (e.g. record videos, reward logs, etc.)
These can the be loaded in the user interface for studies
"""
import argparse
import asyncio
import os
import sys

from rlhfblender.data_collection import framework_selector as framework_selector
from rlhfblender.utils.data_generation import generate_data

if __name__ == "__main__":
    # must fit:     python rlhfblender.generate_data --exp_name MyExperiment --env_id MyEnv-v0 --model_path path/to/model.zip --num_episodes 100 --num_parallel 10

    parser = argparse.ArgumentParser(description="Generate data for RLHFBlender")
    parser.add_argument("--exp", type=str, help="The experiment name.")
    parser.add_argument("--env", type=str, help="The environment id.")
    #parser.add_argument("--model-path", type=str, help="The path to the model.", default="")
    #parser.add_argument("--random", type=bool, help="Whether to use random action selection.", default=False)
    parser.add_argument("--num-episodes", type=int, help="The number of episodes to run.", default=10)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", action="store_true")
    group.add_argument("--model-path", type=str, default="")

    parser.add_argument("--checkpoints", type=str, help="Interpreted as a list of ints, e.g. 0,1,2,3,4,5", default="0")

    args = parser.parse_args()

    # make sure that env and exp are set
    if args.exp is None or args.env is None:
        print("Please specify an environment and an experiment to generate data for.")
        sys.exit(1)

    if args.random:
        args.model_path = ""
        checkpoints = "-1"
    else:
        checkpoints = [int(x) for x in args.checkpoints.split(",")]

    benchmark_dicts = [
        {
            "env_id": args.env,
            "benchmark_type": "trained" if not args.random else "random",
            "benchmark_id": args.exp,
            "checkpoint_step": checkpoints,
            "n_episodes": args.num_episodes,
            "path": args.model_path,
        }
    ]

    asyncio.run(generate_data(benchmark_dicts))

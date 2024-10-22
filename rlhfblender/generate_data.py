"""
Pre-generated data for the RLHFBlender project (e.g. record videos, reward logs, etc.)
These can the be loaded in the user interface for studies
"""

import argparse
import asyncio
import os
import importlib
import traceback 
import sys

from rlhfblender.data_collection import framework_selector as framework_selector
from rlhfblender.utils.data_generation import generate_data, register_env, register_experiment, init_db
from rlhfblender.utils import process_env_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for RLHFBlender")
    parser.add_argument("--exp", type=str, help="The experiment name.", default="")
    parser.add_argument("--env", type=str, help="The environment id.", default="", required=True)
    parser.add_argument("--num-episodes", type=int, help="The number of episodes to run.", default=10)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", action="store_true", help="Use random agent")
    group.add_argument("--model-path", type=str, default="", help="Path to the trained model")
    group.add_argument("--model-base-path", 
                       type=str, 
                       default="rlhfblender_demo_models",
                       help="Base path to the trained model checkpoints")
    
    parser.add_argument("--checkpoints", type=str, nargs="+", default=["-1"], help="The checkpoint steps to use.")

    parser.add_argument(
        "--project",
        type=str,
        help="(Optional) The project name. Defaults to RLHF-Blender.",
        default="RLHF-Blender",
    )

    # Args for env registration
    parser.add_argument(
        "--env-gym-entrypoint",
        type=str,
        help="(Optional) The gym entry point for the environment.",
        default="",
    )
    parser.add_argument(
        "--env-display-name",
        type=str,
        help="(Optional) The display name for the environment.",
        default="",
    )
    parser.add_argument(
        "--additional-gym-packages",
        type=str,
        nargs="+",
        help="(Optional) Additional gym packages to import.",
        default=[],
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        help='Environment Kwargs (e.g. --env-kwargs key1:value1 key2:value2), e.g.: "env_wrapper:stable_baselines3.common.atari_wrappers.AtariWrapper frame_stack:4"',
        default=[],
    )
    parser.add_argument(
        "--framework",
        type=str,
        help="(Optional) The framework used for training. Defaults to StableBaselines3.",
        default="StableBaselines3",
    )
    parser.add_argument(
        "--env-description",
        type=str,
        help="(Optional) A description for the environment.",
        default="",
    )

    args = parser.parse_args()

    if args.env == "":
        print("Please specify an environment to generate data for.")
        sys.exit(1)

    # Parse env_kwargs
    env_kwargs = {}
    if args.env_kwargs:
        if args.exp == "":
            print("Please specify an experiment name if you want to register environment kwargs.")
            sys.exit(1)
        # turn into dict
        for kwarg in args.env_kwargs:
            key, value = kwarg.split(":")
            env_kwargs[key] = value

    # Initialize database
    asyncio.run(init_db())

    # Register environment if necessary
    asyncio.run(
        register_env(
            env_id=args.env,
            entry_point=args.env_gym_entrypoint,
            display_name=args.env_display_name,
            additional_gym_packages=args.additional_gym_packages,
            env_kwargs=env_kwargs,
            env_description=args.env_description,
            project=args.project,
        )
    )

    if args.random:
        args.model_path = ""
        checkpoints = ["-1"]
    else:
        checkpoints = args.checkpoints

    if not args.random:
        model_path = args.model_path if args.model_path != "" else os.path.join(args.model_base_path, process_env_name(args.env))
    else:
        model_path = args.model_path # random agent does not need a model path

    # Register experiment if necessary
    if args.exp != "":
        args.exp = f"{args.env}_{'random' if args.random else 'trained'}_experiment"
        asyncio.run(
            register_experiment(
                exp_name=args.exp,
                env_id=args.env,
                env_kwargs=env_kwargs,
                path=model_path,
                framework="random" if args.random else args.framework,
                project=args.project,
            )
        )

    benchmark_dicts = [
        {
            "env": args.env,
            "benchmark_type": "random" if args.random else "trained",
            "exp": args.exp,
            "checkpoint_step": checkpoint,
            "n_episodes": args.num_episodes,
            "path": model_path,
            "framework": "random" if args.random else args.framework,
        }
        for checkpoint in checkpoints
    ]

    try:
        asyncio.run(generate_data(benchmark_dicts))
    except Exception as e:
        print(f"Error: {e} - Did not generate data.")
        # print stacktrace
        traceback.print_exc()
    finally:
        print("Data generation finished.")
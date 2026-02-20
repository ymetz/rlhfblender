import argparse
import os
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

yaml.add_multi_constructor("!!", lambda loader, suffix, node: None)
yaml.add_multi_constructor(
    "tag:yaml.org,2002:python/name",
    lambda loader, suffix, node: None,
    Loader=yaml.SafeLoader,
)


def collect_results(
    model_base_path: str = "gt_agents",  # Updated default path
    algorithms: List[str] = ["ppo", "sac"],
    output_dir: Optional[str] = None,
    full_param_df: bool = False,
) -> pd.DataFrame:
    """
    Collect and process results from multiple algorithm runs.

    Args:
        model_base_path: Base path for model results
        algorithms: List of algorithms to process
        output_dir: Directory to save results (if None, uses model_base_path)

    Returns:
        DataFrame containing processed results
    """
    if not os.path.isabs(model_base_path):
        model_base_path = os.path.abspath(model_base_path)

    # Process each algorithm
    all_results = pd.DataFrame()
    if len(algorithms) > 0:
        for algorithm in algorithms:
            print(f"Processing algorithm: {algorithm}")
            algorithm_df = process_algorithm_results(model_base_path, algorithm, full_param_df=full_param_df)
            all_results = pd.concat([all_results, algorithm_df], ignore_index=True)
    else:
        algorithm_df = process_algorithm_results(model_base_path, None, full_param_df=full_param_df)
        all_results = pd.concat([all_results, algorithm_df], ignore_index=True)

    # Sort and save results
    if not all_results.empty:
        all_results = all_results.sort_values(["environment", "algorithm"])
        if output_dir is None:
            output_dir = model_base_path
        else:
            output_dir = os.path.abspath(output_dir)
        output_path = os.path.join(output_dir, "collected_results.csv")
        all_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("No results were collected!")

    return all_results


def process_algorithm_results(model_base_path: str, algorithm: str = None, full_param_df: bool = False) -> pd.DataFrame:
    """Process results for a single algorithm."""
    df = pd.DataFrame()
    if algorithm is not None:
        algorithm_path = os.path.join(model_base_path, algorithm)
    else:
        algorithm_path = model_base_path
    if not os.path.exists(algorithm_path):
        print(f"Path not found for algorithm {algorithm}: {algorithm_path}")
        return df

    for run_dir in os.listdir(algorithm_path):
        if "." in run_dir:
            # for ipython-checkpoints etc.
            continue
        env = run_dir.split("_")[0]
        full_run_dir = os.path.join(algorithm_path, run_dir)
        try:
            # Read arguments and config
            args_path = os.path.join(full_run_dir, env, "args.yml")
            config_path = os.path.join(full_run_dir, env, "config.yml")
            if not os.path.isfile(args_path):
                print(f"Args file not found: {args_path}")
                continue

            # Read YAML files
            with open(args_path) as theyaml:
                next(theyaml)  # Skip first line
                run_arguments = yaml.safe_load(theyaml)
            with open(config_path) as theyaml:
                next(theyaml)  # Skip first line
                config = yaml.safe_load(theyaml)

            # Combine all parameters
            the_dict = {}
            for elem in run_arguments[0]:
                the_dict[elem[0]] = elem[1]
            for elem in config[0]:
                the_dict[elem[0]] = elem[1]

            # Load evaluation results
            eval_path = os.path.join(full_run_dir, "evaluations.npz")
            evals = np.mean(np.load(eval_path)["results"][-1])

            # Create result dictionary
            if not full_param_df:
                result_dict = {
                    "algorithm": the_dict["algorithm"] if "algorithm" in the_dict else the_dict["algo"],
                    "environment": the_dict["environment"] if "environment" in the_dict else the_dict["env"],
                    "run": run_dir,
                    "seed": the_dict["seed"],
                    "eval_score": evals,
                }
            else:
                result_dict = {
                    **the_dict,
                    "algorithm": the_dict["algorithm"] if "algorithm" in the_dict else the_dict["algo"],
                    "environment": the_dict["environment"] if "environment" in the_dict else the_dict["env"],
                    "run": run_dir,
                    "seed": the_dict["seed"],
                    "eval_score": evals,
                }

            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
        except Exception as e:
            print(f"Error processing {full_run_dir}:")
            print(traceback.format_exc())
            continue
    return df


def main():
    parser = argparse.ArgumentParser(
        prog="CollectResults",
        description="Collecting results from multiple algorithm runs",
    )
    parser.add_argument("--model-base-path", default="train_baselines/gt_agents")  # Updated default path
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ppo", "sac"],
        help="List of algorithms to process (default: ppo sac)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: same as model-base-path)",
    )

    args = parser.parse_args()
    collect_results(args.model_base_path, args.algorithms, args.output_dir)


if __name__ == "__main__":
    main()

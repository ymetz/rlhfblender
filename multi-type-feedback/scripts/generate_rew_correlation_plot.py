import argparse
import os
import pickle
import pickle as pkl
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.cluster import MiniBatchKMeans

from multi_type_feedback.networks import LightningNetwork


def generate_correlation_data(
    env, algo, seed=6389, reward_seed=1789, num_samples=int(1e4), noise_level=0.0
):
    with open(f"samples/{algo}_{env}_{seed}.pkl", "rb") as file:
        data = pkl.load(file)

    all_in = []
    all_rews = []
    for idx, seg in enumerate(data["segments"]):
        all_in.extend([(s[0].squeeze(), s[1]) for s in seg])
        all_rews.extend([s[2] for s in seg])
    all_rews = np.array(all_rews)

    indices = random.sample(range(len(all_in)), num_samples)

    # Sample from both lists using the same indices
    input_data = [all_in[i] for i in indices]
    rews = all_rews[indices]

    rew_fn_types = [
        "evaluative",
        "comparative",
        "demonstrative",
        "corrective",
        "descriptive",
        "descriptive_preference",
    ]
    reward_functions = []

    base_dir = "reward_models_lul"
    rew_functions = []
    for type in rew_fn_types:
        if noise_level > 0.0:
            rew_functions.append(
                os.path.join(
                    base_dir,
                    f"{algo}_{env}_{reward_seed}_{type}_{reward_seed}_noise_{noise_level}.ckpt",
                )
            )
        else:
            rew_functions.append(
                os.path.join(
                    base_dir, f"{algo}_{env}_{reward_seed}_{type}_{reward_seed}.ckpt"
                )
            )

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    def reward_fn(reward_model_path):
        return lambda input: LightningNetwork.load_from_checkpoint(
            reward_model_path, map_location=device
        )(
            torch.as_tensor(
                np.array([input[0]] * 4), device=device, dtype=torch.float
            ).unsqueeze((1)),
            torch.as_tensor(
                np.array([input[1]] * 4), device=device, dtype=torch.float
            ).unsqueeze(1),
        )

    n_functions = len(reward_functions)

    # Compute rewards for all functions
    pred_rewards = []
    pred_std = []
    for i, path in enumerate(rew_functions):
        func = reward_fn(path)
        with torch.no_grad():
            preds = torch.vstack([func(x).squeeze() for x in input_data])
            pred_rewards.append(torch.mean(preds, axis=1).cpu().numpy())
            pred_std.append(torch.mean(preds, axis=1).cpu().numpy())
            print(f"Finished {i+1}/{len(rew_functions)} rew. function")

    in_names = ["Ground Truth"] + rew_fn_types

    pred_rewards = [rews] + [pr.squeeze() for pr in pred_rewards]

    with open(
        os.path.join(
            "correlation_data",
            f"corr_{env}_{algo}_noise_{noise_level}_{reward_seed}.pkl",
        ),
        "wb",
    ) as feedback_file:
        pickle.dump(pred_rewards, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
    parser.add_argument(
        "--environment", type=str, default="HalfCheetah-v5", help="Environment"
    )
    parser.add_argument(
        "--n-feedback",
        type=int,
        default=int(10000),
        help="How many feedback instances should be generated",
    )
    parser.add_argument(
        "--seed", type=int, default=1789, help="TODO: Seed for env and stuff"
    )
    parser.add_argument("--noise-level", type=float, default=0.0)

    args = parser.parse_args()

    generate_correlation_data(
        args.environment,
        args.algorithm,
        reward_seed=args.seed,
        num_samples=args.n_feedback,
        noise_level=args.noise_level,
    )

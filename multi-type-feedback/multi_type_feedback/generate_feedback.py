import bisect
import itertools
import os
import pickle
import random
import re
import tempfile
import warnings
from pathlib import Path
from typing import List, Type, Union

import gymnasium as gym
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import Tensor

from multi_type_feedback.utils import TrainingUtils

try:
    from train_baselines.benchmark_evals import collect_results
except ImportError:
    collect_results = None


def predict_expert_value(expert_model: Union[PPO, SAC], observation: np.ndarray, actions: Tensor = None) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""

    expert_model, norm_env = expert_model

    if norm_env is not None:
        observation = norm_env.normalize_obs(observation)

    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.min(
            (
                torch.cat(expert_model.policy.critic_target(observation, actions), dim=1)
                if isinstance(expert_model, SAC)
                else expert_model.policy.predict_values(observation)
            ),
            dim=1,
            keepdim=True,
        )[0]


def one_hot_vector(k, max_val):
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec


def get_state_dimensions(segments):
    """Get dimensions from first state to initialize arrays."""
    first_seg = segments[0]
    first_state = first_seg[0]
    state_dim = len(
        np.concatenate(
            (
                first_state[0].squeeze(0).flatten(),
                (np.expand_dims(first_state[1], 0) if first_state[1].ndim == 0 else first_state[1]),
            )
        )
    )
    return state_dim


def count_total_states(segments):
    """Count total number of states across all segments."""
    return sum(len(seg) for seg in segments)


def batch_generator(segments, batch_size):
    """Generate batches of states and rewards from segments."""
    current_batch_states = []
    current_batch_rewards = []

    for seg in segments:
        for state in seg:
            # Process single state
            obs = np.concatenate(
                (
                    state[0].squeeze(0).flatten(),
                    np.expand_dims(state[1], 0) if state[1].ndim == 0 else state[1],
                )
            )
            reward = state[2]

            current_batch_states.append(obs)
            current_batch_rewards.append(reward)

            if len(current_batch_states) == batch_size:
                yield np.array(current_batch_states), np.array(current_batch_rewards)
                current_batch_states = []
                current_batch_rewards = []

    # Return any remaining states
    if current_batch_states:
        yield np.array(current_batch_states), np.array(current_batch_rewards)


def memory_efficient_clustering(segments, n_feedback):
    """Perform memory-efficient clustering on segments."""
    # Initialize KMeans
    state_dim = get_state_dimensions(segments)
    total_states = count_total_states(segments)
    kmeans = MiniBatchKMeans(n_clusters=n_feedback, batch_size=n_feedback, random_state=42)

    # First pass: Train KMeans
    print("Training KMeans...")
    for states_batch, _ in batch_generator(segments, n_feedback):
        kmeans.partial_fit(states_batch)

    # Initialize cluster statistics
    cluster_sums = np.zeros((n_feedback, state_dim))
    cluster_reward_sums = np.zeros(n_feedback)
    cluster_counts = np.zeros(n_feedback)

    # Second pass: Compute cluster statistics
    print("Computing cluster statistics...")
    for states_batch, rewards_batch in batch_generator(segments, n_feedback):
        # Get cluster assignments for this batch
        batch_clusters = kmeans.predict(states_batch)

        # Update cluster statistics
        for i in range(n_feedback):
            mask = batch_clusters == i
            if np.any(mask):
                cluster_sums[i] += np.sum(states_batch[mask], axis=0)
                cluster_reward_sums[i] += np.sum(rewards_batch[mask])
                cluster_counts[i] += np.sum(mask)

    # Compute final cluster representatives and rewards
    cluster_representatives = []
    cluster_rewards = []

    for i in range(n_feedback):
        if cluster_counts[i] > 0:
            rep = cluster_sums[i] / cluster_counts[i]
            reward = cluster_reward_sums[i] / cluster_counts[i]
            if not np.any(np.isnan(rep)):
                cluster_representatives.append(rep)
                cluster_rewards.append(reward)

    return (np.array(cluster_representatives), np.array(cluster_rewards), kmeans)


def create_segments(arr, start_indices, done_indices, segment_length):
    """
    Creates array segments with target length (segment_length) and minimum length min_segment_len,
    selects the longest contiguous array within a segment [start_indices: start_indeces+segment_length]
    """
    segments = []

    for start in start_indices:
        end = start + segment_length

        # Find the position of the first done_index greater than or equal to start
        insert_pos = bisect.bisect_left(done_indices, start)

        # Collect all done_indices within the range [start, end)
        relevant_done_indices = []
        while insert_pos < len(done_indices) and done_indices[insert_pos] < end:
            relevant_done_indices.append(done_indices[insert_pos])
            insert_pos += 1

        # If there are no relevant done_indices, take the full segment
        if not relevant_done_indices:
            segment = arr[start:end]
        else:
            # Consider the segment before the first done_index
            longest_segment = arr[start : relevant_done_indices[0]]

            # Consider segments between each pair of consecutive done_indices
            for i in range(len(relevant_done_indices) - 1):
                segment = arr[relevant_done_indices[i] : relevant_done_indices[i + 1]]
                if len(segment) > len(longest_segment):
                    longest_segment = segment

            # Consider the segment after the last done_index
            segment = arr[relevant_done_indices[-1] : end]
            if len(segment) > len(longest_segment):
                longest_segment = segment

            # Use the longest valid segment
            segment = longest_segment

        segments.append(segment)

    return segments


def discounted_sum_numpy(rewards, gamma):
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))


def equal_depth_binning_with_indices(data, num_bins):
    data = np.array(data)
    # Sort the data and get the original indices
    sorted_indices = np.argsort(data)
    sorted_data = np.sort(data)

    # Determine the number of elements per bin
    bin_size = len(data) // num_bins
    remainder = len(data) % num_bins

    bins = []
    bin_indices = np.zeros(len(data), dtype=int)
    start = 0

    for i in range(num_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        bin_indices[sorted_indices[start:end]] = i
        bins.append(sorted_data[start:end])
        start = end

    return bin_indices, bins


def equal_width_binning_with_indices(data, num_bins):
    data = np.array(data)
    # Find the minimum and maximum values in the data
    min_val, max_val = np.min(data), np.max(data)

    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Use numpy's digitize function to assign bin indices
    bin_indices = np.digitize(data, bin_edges[:-1])

    # Create the bins
    bins = [data[(bin_indices == i)] for i in range(1, num_bins + 1)]

    return bin_indices, bins


def get_preference_pairs(segments, opt_gaps, n_feedback, tolerance=0.25):
    all_pairs = list(enumerate(itertools.combinations(range(len(segments)), 2)))
    random.shuffle(all_pairs)

    preferences = []
    sampled_indices = []

    for idx, (a, b) in all_pairs:
        gap_diff = opt_gaps[a] - opt_gaps[b]
        if abs(gap_diff) > tolerance:
            if gap_diff > 0:
                preferences.append((a, b, 1))
            else:
                preferences.append((b, a, 1))
            sampled_indices.append(idx)

            if len(preferences) == n_feedback:
                break

    if len(preferences) < n_feedback:
        raise ValueError(
            f"Could only generate {len(preferences)} preferences with the given tolerance. Increase the number of segments, decrease the tolerance, or decrease n_feedback."
        )

    return preferences


def get_preference_pairs_descript(clusters, rews, n_feedback, tolerance=0.01):
    all_pairs = list(enumerate(itertools.combinations(range(len(clusters)), 2)))
    random.shuffle(all_pairs)

    preferences = []
    sampled_indices = []

    for idx, (a, b) in all_pairs:
        rew_diff = rews[a] - rews[b]
        if abs(rew_diff) > tolerance:
            if rew_diff < 0:
                preferences.append((a, b, 1))
            else:
                preferences.append((b, a, 1))
            sampled_indices.append(idx)

            if len(preferences) == n_feedback:
                break

    if len(preferences) < n_feedback:
        print(
            f"Could only generate {len(preferences)} preferences with the given tolerance. Increase the number of segments, decrease the tolerance, or decrease n_feedback."
        )

    return preferences


def debug_feedback_output(feedback_data):
    print("\nDebugging Feedback Output:")
    print(f"Number of segments: {len(feedback_data['segments'])}")
    print(f"Number of ratings: {len(feedback_data['ratings'])}")
    print(f"Number of preferences: {len(feedback_data['preferences'])}")
    print(f"Number of demos: {len(feedback_data['demos'])}")
    print(f"Number of corrections: {len(feedback_data['corrections'])}")
    print(f"Number of cluster descriptions: {len(feedback_data['description'])}")
    print(f"Number of description preferences: {len(feedback_data['description_preference'])}")
    print(f"Number of optimality gaps: {len(feedback_data['opt_gaps'])}")

    # Additional checks
    print("\nConsistency Checks:")
    n_feedback = len(feedback_data["segments"])
    print(
        f"All main feedback arrays have {n_feedback} elements: {all(len(feedback_data[key]) == n_feedback for key in ['segments', 'ratings', 'demos', 'corrections', 'opt_gaps'])}"
    )

    # Check segment lengths
    segment_lengths = [len(segment) for segment in feedback_data["segments"]]
    print(f"Minimum segment length: {min(segment_lengths)}")
    print(f"Maximum segment length: {max(segment_lengths)}")
    print(f"Average segment length: {sum(segment_lengths) / len(segment_lengths):.2f}")


def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC], Union[VecNormalize, None]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v5",
    checkpoints_path: str = "gt_agents",
    total_steps_factor: int = 50,
    n_feedback: int = 100,
    segment_len: int = 50,
    oversampling_factor: int = 1.5,
    min_segment_len: int = 25,
    algorithm: str = "sac",
    device: str = "cuda",
    action_one_hot: bool = False,
    binning_type: str = "width",
) -> dict:
    """Generate agent's observations and feedback in the training environment."""
    feedback_id = f"{algorithm}_{environment_name.replace('/', '-')}"

    print(os.listdir(os.path.join(checkpoints_path, algorithm)))
    possible_checkpoint_indices = [
        str(model_dir.split("_")[-1])
        for model_dir in os.listdir(os.path.join(checkpoints_path, algorithm))
        if f"{environment_name.replace('/', '-')}" in model_dir
    ]
    checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{environment_name.replace('/', '-')}_1")

    print(f"Generating feedback for: {feedback_id}")

    # Adaptive oversampling
    oversampling_factor = oversampling_factor
    target_n_feedback = int(n_feedback * oversampling_factor)

    checkpoint_files = [file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)] or [
        f"{environment_name}.zip"
    ]

    total_steps = n_feedback * total_steps_factor
    num_checkpoints = len(checkpoint_files) + 1
    steps_per_checkpoint = total_steps // num_checkpoints
    feedback_per_checkpoint = target_n_feedback // num_checkpoints
    gamma = expert_models[0][0].gamma

    checkpoint_files = ["random"] + sorted(checkpoint_files, key=lambda x: int(re.search(r"\d+", x).group()))

    if action_one_hot:
        one_hot_dim = environment.action_space.n  # only works for discrete spaces

    print(
        f"""
    Feedback Generation Debug Info:
      Feedback ID: {feedback_id}
      Checkpoints Directory: {checkpoints_dir}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Target Feedback: {n_feedback}
      Oversampled Generated Feedback Instances: {target_n_feedback}
      Feedback per Checkpoint: {feedback_per_checkpoint}
      Env. Gamma: {gamma}
    """
    )

    segments = []
    state_copies = []
    for model_file in checkpoint_files:
        feedback = []
        fb_indices = random.sample(range(steps_per_checkpoint - segment_len + 1), k=feedback_per_checkpoint + 1)
        final_segment_indices = sorted(set(fb_indices))

        if model_file != "random":
            # replace the _1 index by other possible indices, this only works if all models have exactly the same number of checkpoints
            model_path = checkpoints_dir.replace("_1", f"_{random.choice(possible_checkpoint_indices)}")
            model = model_class.load(
                os.path.join(model_path, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
            norm_env_path = os.path.join(model_path, environment_name, "vecnormalize.pkl")
            norm_env = (
                VecNormalize.load(norm_env_path, DummyVecEnv([lambda: environment])) if os.path.isfile(norm_env_path) else None
            )
        else:
            model = None
            norm_env = None

        observation, _ = environment.reset()

        for step in range(steps_per_checkpoint):
            if step in final_segment_indices:
                state_copies.append(environment.save_state(observation=observation))

            if model is not None:
                action, _ = model.predict(
                    norm_env.normalize_obs(observation) if norm_env else observation,
                    deterministic=True,
                )
            else:
                action = environment.action_space.sample()

            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            if action_one_hot:
                action = one_hot_vector(action, one_hot_dim)

            feedback.append((np.expand_dims(observation, axis=0), action, reward, done))

            observation = next_observation if not done else environment.reset()[0]

        segments.extend(
            create_segments(
                feedback,
                final_segment_indices,
                np.where([f[3] for f in feedback])[0],
                segment_len,
            )
        )

        print(
            f"Generated segments: {len(segments)} of target {target_n_feedback} (Oversampling Factor: {oversampling_factor})"
        )

    opt_gaps = []
    for seg in segments:
        """
        initial_vals = [predict_expert_value(expert_model, np.array(seg[0][0])).item() for expert_model in expert_models]
        initial_val = np.mean(initial_vals)
        #initial_val = initial_vals[0]

        discounted_rew_sum = discounted_sum_numpy([s[2] for s in seg[:-1]], gamma)

        final_vals = [predict_expert_value(expert_model, np.array(seg[-1][0])).item() for expert_model in expert_models]
        final_val = np.mean(final_vals)
        #final_val = final_vals[0]

        opt_gap = initial_val - (discounted_rew_sum + gamma ** len(seg) * final_val)
        """
        opt_gap = -discounted_sum_numpy([s[2] for s in seg], gamma)
        opt_gaps.append(opt_gap)

    max_rating = 10
    ratings = max_rating - (
        equal_width_binning_with_indices(opt_gaps, max_rating)[0]
        if binning_type == "width"
        else equal_depth_binning_with_indices(opt_gaps, max_rating)[0]
    )

    print("[INFO] Successfully generated evaluative feedback")

    demos = []
    corrections = []
    improvements = []

    for i, state in enumerate(state_copies):
        current_demos = []
        current_expert_model_returns = []

        for exp_model_index, (expert_model, exp_norm_env) in enumerate(expert_models):
            _, _ = environment.reset()
            obs = environment.load_state(state)

            demo = []
            for _ in range(segment_len):
                action, _ = expert_model.predict(
                    exp_norm_env.normalize_obs(obs) if exp_norm_env else obs,
                    deterministic=True,
                )
                new_obs, rew, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated

                if action_one_hot:
                    action = one_hot_vector(action, one_hot_dim)

                demo.append((np.expand_dims(obs, axis=0), action, rew, done, exp_model_index))
                obs = new_obs

                if done:
                    break

            current_demos.append(demo)
            current_expert_model_returns.append(discounted_sum_numpy([d[2] for d in demo], gamma))

        best_index = np.argmax(current_expert_model_returns)
        best_demo = current_demos[best_index]
        best_demo_return = current_expert_model_returns[best_index]

        original_return = discounted_sum_numpy([s[2] for s in segments[i]], gamma)

        if best_demo_return > original_return:
            demos.append(best_demo)
            corrections.append((segments[i], best_demo))
            improvements.append(best_demo_return - original_return)
        else:
            # if the correction is worse..turn it around
            demos.append(segments[i])
            corrections.append((best_demo, segments[i]))
            improvements.append(original_return - best_demo_return)

        if i % 100 == 0:
            print(f"Generated demos: {len(demos)} of target {target_n_feedback}")

    sorted_indices = np.argsort(improvements)[::-1]

    final_demos = []
    final_corrections = []
    selected_indices = []
    for idx in sorted_indices:
        if len(final_demos) >= n_feedback:
            break
        if corrections[idx] is not None:
            final_demos.append(demos[idx])
            final_corrections.append(corrections[idx])
            selected_indices.append(idx)

    if len(final_demos) < n_feedback:
        for idx in sorted_indices:
            if len(final_demos) >= n_feedback:
                break
            if idx not in selected_indices and demos[idx] is not None:
                final_demos.append(demos[idx])
                final_corrections.append((segments[idx], demos[idx]))
                selected_indices.append(idx)

    # Use selected_indices to sample segments, ratings, and opt_gaps
    segments = [segments[i] for i in selected_indices]
    ratings = [ratings[i] for i in selected_indices]
    opt_gaps = [opt_gaps[i] for i in selected_indices]

    # now we can sample the pairs after we have pruned segments
    tolerance = np.std(opt_gaps) / 10.0
    preferences = get_preference_pairs(segments, opt_gaps, n_feedback, tolerance=tolerance)

    print("[INFO] Successfully generated comparative feedback")

    demos = final_demos
    corrections = final_corrections

    print("[INFO] Successfully generated demonstrative/corrective feedback")

    demo_data = {"demos": demos, "corrections": corrections}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(demo_data, tmp)
        demo_file = tmp.name

    # Remove demos and corrections from memory - they'll be garbage collected
    del demos
    del corrections
    del demo_data

    cluster_representatives, cluster_rewards, kmeans = memory_efficient_clustering(segments=segments, n_feedback=n_feedback)

    obs_dim = np.prod(segments[0][0][0].squeeze(0).shape)
    cluster_descriptions = [
        (rep[:obs_dim], rep[obs_dim:], reward) for rep, reward in zip(cluster_representatives, cluster_rewards)
    ]
    tolerance = np.std(cluster_rewards) / 10.0
    descr_preferences = get_preference_pairs_descript(cluster_descriptions, cluster_rewards, n_feedback, tolerance=tolerance)

    # After clustering is done, restore demos and corrections
    with open(demo_file, "rb") as f:
        demo_data = pickle.load(f)
    os.unlink(demo_file)  # Clean up temporary file

    # Prepare final feedback dictionary
    return {
        "segments": segments,
        "ratings": ratings,
        "preferences": preferences,
        "demos": demo_data["demos"],
        "corrections": demo_data["corrections"],
        "description": cluster_descriptions,
        "description_preference": descr_preferences,
        "opt_gaps": opt_gaps,
    }


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--n-steps-factor",
        type=int,
        default=20,
        help="Number of steps sampled for each feedback instance",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=50,
        help="Segment length for feedback generation",
    )
    parser.add_argument("--min-segment-len", type=int, default=None, help="Minimum segment length")
    parser.add_argument("--oversampling_factor", type=float, default=1.5, help="Oversampling factor")
    parser.add_argument("--save-folder", type=str, default="feedback", help="Save folder")
    parser.add_argument("--top-n-models", type=int, default=3, help="Top N models to use")
    parser.add_argument(
        "--expert-model-base-path",
        type=str,
        default="gt_agents",
        help="Expert model base path",
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    device = TrainingUtils.get_device()

    feedback_id, _ = TrainingUtils.get_model_ids(args)
    feedback_path = Path(args.save_folder) / f"{feedback_id}.pkl"

    environment = TrainingUtils.setup_environment(
        args.environment, args.seed, env_kwargs=TrainingUtils.parse_env_kwargs(args.environment_kwargs)
    )

    # try to load most recent benchmark scores for expert models, works if experts were created via train_baselines
    # scripts
    if collect_results is not None:
        try:
            collect_results(
                args.expert_model_base_path.replace("\\", "/"),
                [args.algorithm],
                str(args.expert_model_base_path),
            )
        except:
            warnings.warn(
                """No expert benchmark results could be found. Only random policies are available. Make sure to train expert models with train_baselines,
                          or change the path. Experts need to be trained with an SB3 MonitorWrapper and EvalCallback to retrieve benchmark scores"""
            )

    expert_models = TrainingUtils.load_expert_models(
        args.environment,
        args.algorithm,
        str(args.expert_model_base_path),
        environment,
        args.top_n_models,
    )

    feedback = generate_feedback(
        model_class=PPO if args.algorithm == "ppo" else SAC,
        expert_models=expert_models,
        environment=environment,
        environment_name=args.environment,
        checkpoints_path=str(args.expert_model_base_path),
        total_steps_factor=args.n_steps_factor,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        oversampling_factor=args.oversampling_factor,
        min_segment_len=args.min_segment_len or args.segment_len // 2,
        algorithm=args.algorithm,
        device=device,
        action_one_hot=isinstance(environment.action_space, gym.spaces.Discrete),
    )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as f:
        pickle.dump(feedback, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

import os
from typing import Any, Tuple
import pickle

import numpy as np
from rlhfblender.data_models.feedback_models import Segment, StandardizedFeedback, State, Target
from rlhfblender.utils import process_env_name

DATA_BASE_PATH = "data"  # Base path for data files


class FeedbackDatasetAdapter:

    @staticmethod
    def _load_npz_trajectory(file_path: str) -> list[Tuple[np.ndarray, np.ndarray, float]]:
        """Load a trajectory stored as an NPZ file with obs/actions[/rewards] arrays."""
        if not file_path:
            return []

        candidate_paths: list[str] = [file_path]
        if not os.path.isabs(file_path):
            candidate_paths.append(os.path.abspath(file_path))
            # Allow paths relative to the data directory
            candidate_paths.append(os.path.join(DATA_BASE_PATH, file_path))

        resolved_path = None
        for candidate in candidate_paths:
            normalized = os.path.normpath(candidate)
            if os.path.exists(normalized):
                resolved_path = normalized
                break

        if resolved_path is None:
            print(f"Warning: Trajectory file not found for path {file_path}")
            return []

        try:
            with np.load(resolved_path, allow_pickle=True) as data:
                obs = data.get("obs")
                actions = data.get("actions")
                rewards = data.get("rewards")

                if obs is None or actions is None:
                    print(f"Warning: Missing obs/actions arrays in {resolved_path}")
                    return []

                obs_len = len(obs)
                if rewards is None or len(rewards) < obs_len:
                    rewards = np.zeros(obs_len, dtype=float)

                trajectory: list[Tuple[np.ndarray, np.ndarray, float]] = []
                for idx in range(obs_len):
                    reward = float(rewards[idx]) if idx < len(rewards) else 0.0
                    trajectory.append((np.asarray(obs[idx]), np.asarray(actions[idx]), reward))

                return trajectory

        except Exception as exc:
            print(f"Warning: Failed to load trajectory from {resolved_path}: {exc}")
            return []

    @staticmethod
    def _gather_target_steps(targets: list[Target]) -> list[Tuple[np.ndarray, np.ndarray, float]]:
        """Collect all step tuples (obs, action, reward) from the provided targets."""
        gathered: list[Tuple[np.ndarray, np.ndarray, float]] = []
        for target in targets:
            segment = FeedbackDatasetAdapter._extract_segment(target)
            if segment:
                gathered.extend(segment)
        return gathered

    @staticmethod
    def _compute_cluster_representative(targets: list[Target]) -> Tuple[np.ndarray, np.ndarray] | None:
        """Compute a representative state-action pair for a cluster of targets."""
        steps = FeedbackDatasetAdapter._gather_target_steps(targets)
        if not steps:
            return None

        try:
            obs_stack = np.stack([np.asarray(step[0]) for step in steps], axis=0)
            action_stack = np.stack([np.asarray(step[1]) for step in steps], axis=0)
            mean_obs = obs_stack.mean(axis=0)
            mean_action = action_stack.mean(axis=0)
            return mean_obs, mean_action
        except Exception as exc:
            print(f"Warning: Could not compute cluster representative, using first sample instead ({exc})")
            first_obs, first_action, _ = steps[0]
            return np.asarray(first_obs), np.asarray(first_action)

    @staticmethod
    def convert_to_dynamic_rlhf_format(feedbacks: list[StandardizedFeedback]) -> dict[str, Any]:
        """Convert processed StandardizedFeedback to DynamicRLHF format."""

        # Group feedbacks by type
        grouped = {}
        for fb in feedbacks:
            grouped.setdefault(fb.feedback_type, []).append(fb)

        def empty_feedback_dict():
            return {
                "segments": [],
                "ratings": [],
                "preferences": [],
                "demos": [],
                "corrections": [],
                "description": [],
                "description_preference": [],
                "opt_gaps": [],
            }

        feedback_by_type = {}

        print("Converting feedbacks to DynamicRLHF format:")
        for fb in feedbacks:
            print(f"  Feedback ID: {fb.feedback_id}, Type: {fb.feedback_type}, Granularity: {fb.granularity}, Targets: {len(fb.targets)}")
        print("Grouped feedback types:", {k: len(v) for k, v in grouped.items()})

        # Process evaluative feedback (ratings)
        if "rating" in grouped:
            data = empty_feedback_dict()
            for fb in grouped["rating"]:
                if fb.targets and (segment := FeedbackDatasetAdapter._extract_segment(fb.targets[0])):
                    data["segments"].append(segment)
                    score = float(fb.content.score)
                    data["ratings"].append(score)
                    data["opt_gaps"].append(-score)

            if data["segments"]:
                feedback_by_type["evaluative"] = data

        # Process comparative feedback (ranking/comparison)
        comparison_types = grouped.get("ranking", []) + grouped.get("comparison", [])
        if comparison_types:
            data = empty_feedback_dict()
            segment_map = {}

            # Collect unique segments and create preference pairs
            for fb in comparison_types:
                # Map segments
                for i, target in enumerate(fb.targets):
                    seg_key = str(target.target_id)
                    if seg_key not in segment_map and (segment := FeedbackDatasetAdapter._extract_segment(target)):
                        segment_map[seg_key] = len(data["segments"])
                        data["segments"].append(segment)
                        pref_value = fb.content.preferences[i] if i < len(fb.content.preferences) else 0
                        data["opt_gaps"].append(-float(pref_value))

                # Create preference pairs
                if (
                    len(fb.targets) >= 2
                    and len(fb.content.preferences) >= 2
                    and all(str(fb.targets[j].target_id) in segment_map for j in range(2))
                ):
                    idx1, idx2 = segment_map[str(fb.targets[0].target_id)], segment_map[str(fb.targets[1].target_id)]
                    pref = 1 if fb.content.preferences[0] > fb.content.preferences[1] else 0
                    data["preferences"].append([idx1, idx2, pref])

            if data["segments"]:
                feedback_by_type["comparative"] = data

        # Process demonstration feedback (recorded demos + fallback to targets)
        if "demonstration" in grouped:
            data = empty_feedback_dict()
            for fb in grouped["demonstration"]:
                demo_segments: list[Tuple[np.ndarray, np.ndarray, float]] = []

                demo_path = getattr(fb.content, "path", None)
                if demo_path:
                    demo_segments = FeedbackDatasetAdapter._load_npz_trajectory(demo_path)

                if not demo_segments and fb.targets:
                    extracted = FeedbackDatasetAdapter._extract_segment(fb.targets[0])
                    if extracted:
                        demo_segments = extracted

                if demo_segments:
                    data["demos"].append(demo_segments)
                else:
                    print(f"Warning: Demonstration {fb.feedback_id} has no trajectory data")

            if data["demos"]:
                feedback_by_type["demonstrative"] = data

        # Process correction feedback (pair recorded correction with original episode segment)
        if "correction" in grouped:
            data = empty_feedback_dict()
            for fb in grouped["correction"]:
                corrected_segment: list[Tuple[np.ndarray, np.ndarray, float]] = []
                reference_segment: list[Tuple[np.ndarray, np.ndarray, float]] = []

                # Load corrected trajectory from recorded file if available
                correction_path = getattr(fb.content, "path", None)
                if correction_path:
                    corrected_segment = FeedbackDatasetAdapter._load_npz_trajectory(correction_path)

                # Attempt to gather reference data from provided targets
                if fb.targets:
                    reference_segment = FeedbackDatasetAdapter._gather_target_steps([fb.targets[0]])
                    # If only a specific step was selected, trim the original trajectory accordingly
                    step_idx = getattr(fb.targets[0], "step", None)
                    if (
                        isinstance(step_idx, int)
                        and step_idx >= 0
                        and step_idx < len(reference_segment)
                    ):
                        reference_segment = reference_segment[step_idx:]

                if not reference_segment and len(fb.targets) > 1:
                    reference_segment = FeedbackDatasetAdapter._gather_target_steps([fb.targets[1]])

                if corrected_segment and reference_segment:
                    data["corrections"].append([reference_segment, corrected_segment])
                else:
                    print(
                        f"Warning: Skipping correction {fb.feedback_id} due to missing "
                        f"{'recorded trajectory' if not corrected_segment else 'reference segment'}"
                    )

            if data["corrections"]:
                feedback_by_type["corrective"] = data

        # Process descriptive feedback (feature selection + cluster ratings)
        descriptive_sources = []
        if "feature_selection" in grouped:
            descriptive_sources.append("feature_selection")
        if "cluster_rating" in grouped:
            descriptive_sources.append("cluster_rating")

        if descriptive_sources:
            data = empty_feedback_dict()
            cluster_rewards = []

            # Feature-selection style descriptive feedback (state importance)
            for fb in grouped.get("feature_selection", []):
                if fb.targets and (segment := FeedbackDatasetAdapter._extract_segment(fb.targets[0])) and segment:
                    state, action = np.asarray(segment[0][0]), np.asarray(segment[0][1])
                    importance = getattr(fb.content, "importance", 0.0)
                    if isinstance(importance, (list, tuple, np.ndarray)):
                        reward = float(importance[0]) if importance else 0.0
                    else:
                        try:
                            reward = float(importance)
                        except (TypeError, ValueError):
                            reward = 0.0

                    data["description"].append((state, action, reward))
                    cluster_rewards.append(reward)

            # Cluster ratings (aggregate states within the cluster)
            for fb in grouped.get("cluster_rating", []):
                representative = FeedbackDatasetAdapter._compute_cluster_representative(fb.targets)
                if representative is None:
                    print(f"Warning: Could not compute representative for cluster rating {fb.feedback_id}")
                    continue

                obs_repr, act_repr = representative
                score = getattr(fb.content, "score", 0.0)
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    numeric_score = 0.0

                data["description"].append((obs_repr, act_repr, numeric_score))
                cluster_rewards.append(numeric_score)

            # Create preference pairs
            for i in range(min(len(data["description"]) - 1, 10)):
                if cluster_rewards[i] != cluster_rewards[i + 1]:
                    pref = 1 if cluster_rewards[i] > cluster_rewards[i + 1] else 0
                    data["description_preference"].append([i, i + 1, pref])

            if data["description"]:
                feedback_by_type["descriptive"] = data

        return feedback_by_type

    @staticmethod
    def save_dynamic_rlhf_format(feedbacks: list[StandardizedFeedback], file_path: str) -> bool:
        """
        Convert processed StandardizedFeedback to DynamicRLHF format split by feedback type and save to pickle files.

        Args:
            feedbacks: List of processed StandardizedFeedback objects
            file_path: Base path where to save the pickle files (will add _{feedback_type}.pkl)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to DynamicRLHF format (split by type)
            feedback_by_type = FeedbackDatasetAdapter.convert_to_dynamic_rlhf_format(feedbacks)

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save each feedback type to a separate file
            saved_files = []
            for feedback_type, data in feedback_by_type.items():
                type_file_path = f"{os.path.splitext(file_path)[0]}_{feedback_type}.pkl"

                with open(type_file_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                saved_files.append(type_file_path)
                print(f"Saved {feedback_type} feedback to: {type_file_path}")
                print(
                    f"  {feedback_type} summary: {len(data['segments'])} segments, "
                    f"{len(data['ratings'])} ratings, "
                    f"{len(data['preferences'])} preferences, "
                    f"{len(data['demos'])} demos, "
                    f"{len(data['corrections'])} corrections"
                )

            print(f"\nTotal files saved: {len(saved_files)}")
            return True

        except Exception as e:
            print(f"Error saving DynamicRLHF format feedback: {e}")
            return False

    @staticmethod
    def _extract_segment(target: Target) -> list[Tuple[np.ndarray, np.ndarray, float]]:
        """Extract segment data from target using episode reference information"""

        # If target already has data loaded, use it
        if hasattr(target, "data") and target.data is not None:
            return target.data

        # Load episode data from files using reference information
        if target.env_name and target.benchmark_id >= 0 and target.checkpoint_step >= 0 and target.episode_num >= 0:
            processed_env_name = process_env_name(target.env_name)

            # Build episode path using the reference information
            episode_path = os.path.join(
                DATA_BASE_PATH,  # Updated path based on your directory structure
                "episodes",
                processed_env_name,
                f"{processed_env_name}_{target.benchmark_id}_{target.checkpoint_step}",
                f"benchmark_{target.episode_num}.npz",
            )

            if os.path.exists(episode_path):
                try:
                    episode_data = np.load(episode_path, allow_pickle=True)

                    print("[INFO: extract_segments] Loading episode data from:", episode_path)

                    # Extract observations, actions, rewards
                    obs = episode_data["obs"]
                    actions = episode_data["actions"]
                    rewards = episode_data["rewards"]

                    # Create segment based on target type
                    if isinstance(target, State):
                        # Extract single state-action-reward tuple
                        if 0 <= target.step < len(obs):
                            return [(obs[target.step], actions[target.step], rewards[target.step])]
                        else:
                            return []
                    elif isinstance(target, Segment):
                        # Extract segment from start to end
                        start = max(0, target.start)
                        end = min(len(obs), target.end + 1) if target.end >= 0 else len(obs)
                        segment = []
                        for i in range(start, end):
                            segment.append((obs[i], actions[i], rewards[i]))
                        return segment
                    else:
                        # For Episode or Entire, return full trajectory
                        segment = []
                        for i in range(len(obs)):
                            segment.append((obs[i], actions[i], rewards[i]))
                        return segment

                except Exception as e:
                    print(f"Warning: Could not load episode data from {episode_path}: {e}")
                    return []
            else:
                print(f"Warning: Episode file not found at {episode_path}")
                return []
        else:
            print(
                f"Warning: Incomplete reference information in target: env_name={target.env_name}, benchmark_id={target.benchmark_id}, checkpoint_step={target.checkpoint_step}, episode_num={target.episode_num}"
            )
            return []

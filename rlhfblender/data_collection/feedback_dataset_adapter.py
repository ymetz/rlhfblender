from typing import Any, Tuple

import numpy as np
from multi_type_feedback.feedback_dataset import FeedbackDataset

from rlhfblender.data_models.feedback_models import Segment, StandardizedFeedback, State, Target

DATA_BASE_PATH = "data"  # Base path for data files

class FeedbackDatasetAdapter:
    """Adapts Option 1 feedback objects to work with existing FeedbackDataset"""

    @staticmethod
    def convert_to_legacy_format(feedbacks: list[StandardizedFeedback]) -> dict[str, Any]:
        """Convert Option 1 feedback objects to legacy FeedbackData format"""

        # Group feedbacks by type
        grouped = {}
        for fb in feedbacks:
            if fb.feedback_type not in grouped:
                grouped[fb.feedback_type] = []
            grouped[fb.feedback_type].append(fb)

        # Convert each type
        legacy_data = {}

        if "rating" in grouped:
            segments = []
            ratings = []
            opt_gaps = []  # For noise calculation

            for fb in grouped["rating"]:
                # Extract segment from target
                segment = FeedbackDatasetAdapter._extract_segment(fb.targets[0])
                segments.append(segment)
                ratings.append(fb.content.score)
                # Calculate opt_gap if needed (you might need to adjust this)
                opt_gaps.append(-fb.content.score)  # Placeholder

            legacy_data["segments"] = segments
            legacy_data["ratings"] = ratings
            legacy_data["opt_gaps"] = opt_gaps

        if "ranking" in grouped or "comparison" in grouped:
            preferences = []
            segments = []
            opt_gaps = []

            # First collect all unique segments
            segment_map = {}

            for fb in grouped.get("ranking", []) + grouped.get("comparison", []):
                for i, target in enumerate(fb.targets):
                    segment = FeedbackDatasetAdapter._extract_segment(target)
                    seg_key = str(target.target_id)
                    if seg_key not in segment_map:
                        segment_map[seg_key] = len(segments)
                        segments.append(segment)
                        # Calculate opt_gap (placeholder)
                        opt_gaps.append(-fb.content.preferences[i] if i < len(fb.content.preferences) else 0)

            # Now create preference pairs
            for fb in grouped.get("ranking", []) + grouped.get("comparison", []):
                if len(fb.targets) >= 2:
                    idx1 = segment_map[str(fb.targets[0].target_id)]
                    idx2 = segment_map[str(fb.targets[1].target_id)]
                    # Determine preference (1 if first is better, 0 otherwise)
                    pref = 1 if fb.content.preferences[0] > fb.content.preferences[1] else 0
                    preferences.append([idx1, idx2, pref])

            legacy_data["segments"] = segments
            legacy_data["preferences"] = preferences
            legacy_data["opt_gaps"] = opt_gaps

        if "demonstration" in grouped:
            demos = []
            for fb in grouped["demonstration"]:
                demo = FeedbackDatasetAdapter._extract_segment(fb.targets[0])
                demos.append(demo)
            legacy_data["demos"] = demos

        if "correction" in grouped:
            corrections = []
            for fb in grouped["correction"]:
                if len(fb.targets) >= 2:
                    traj1 = FeedbackDatasetAdapter._extract_segment(fb.targets[0])
                    traj2 = FeedbackDatasetAdapter._extract_segment(fb.targets[1])
                    corrections.append([traj1, traj2])
            legacy_data["corrections"] = corrections

        if "feature_selection" in grouped:
            description = []
            for fb in grouped["feature_selection"]:
                # Extract state-action pair from target
                segment = FeedbackDatasetAdapter._extract_segment(fb.targets[0])
                if segment:
                    state = segment[0][0]  # First state
                    action = segment[0][1]  # First action
                    # Use feature importance as reward proxy
                    reward = fb.content.importance if hasattr(fb.content, "importance") else 0.0
                    description.append([state, action, reward])
            legacy_data["description"] = description

        return legacy_data

    @staticmethod
    def convert_to_dynamic_rlhf_format(feedbacks: list[StandardizedFeedback]) -> dict[str, Any]:
        """Convert processed StandardizedFeedback to DynamicRLHF format."""
        
        # Group feedbacks by type
        grouped = {}
        for fb in feedbacks:
            grouped.setdefault(fb.feedback_type, []).append(fb)

        def empty_feedback_dict():
            return {
                "segments": [], "ratings": [], "preferences": [], "demos": [],
                "corrections": [], "description": [], "description_preference": [], "opt_gaps": []
            }

        feedback_by_type = {}

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
                if (len(fb.targets) >= 2 and len(fb.content.preferences) >= 2 and 
                    all(str(fb.targets[j].target_id) in segment_map for j in range(2))):
                    idx1, idx2 = segment_map[str(fb.targets[0].target_id)], segment_map[str(fb.targets[1].target_id)]
                    pref = 1 if fb.content.preferences[0] > fb.content.preferences[1] else 0
                    data["preferences"].append([idx1, idx2, pref])
            
            if data["segments"]:
                feedback_by_type["comparative"] = data

        # Process other feedback types
        type_mappings = {
            "demonstration": ("demonstrative", "demos", lambda fb: [FeedbackDatasetAdapter._extract_segment(fb.targets[0])] if fb.targets else []),
            "correction": ("corrective", "corrections", lambda fb: [[FeedbackDatasetAdapter._extract_segment(fb.targets[0]), 
                                                                FeedbackDatasetAdapter._extract_segment(fb.targets[1])]] 
                        if len(fb.targets) >= 2 else [])
        }
        
        for fb_type, (output_type, key, extractor) in type_mappings.items():
            if fb_type in grouped:
                data = empty_feedback_dict()
                for fb in grouped[fb_type]:
                    extracted = extractor(fb)
                    data[key].extend([item for item in extracted if item])
                
                if data[key]:
                    feedback_by_type[output_type] = data

        # Process descriptive feedback (feature selection)
        if "feature_selection" in grouped:
            data = empty_feedback_dict()
            cluster_rewards = []
            
            for fb in grouped["feature_selection"]:
                if (fb.targets and (segment := FeedbackDatasetAdapter._extract_segment(fb.targets[0])) and segment):
                    state, action = segment[0][0], segment[0][1]
                    reward = getattr(fb.content, "importance", 0.0)
                    data["description"].append((state, action, reward))
                    cluster_rewards.append(reward)
            
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
        import pickle
        import os
        
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
                print(f"  {feedback_type} summary: {len(data['segments'])} segments, "
                      f"{len(data['ratings'])} ratings, "
                      f"{len(data['preferences'])} preferences, "
                      f"{len(data['demos'])} demos, "
                      f"{len(data['corrections'])} corrections")
            
            print(f"\nTotal files saved: {len(saved_files)}")
            return True
            
        except Exception as e:
            print(f"Error saving DynamicRLHF format feedback: {e}")
            return False

    @staticmethod
    def _extract_segment(target: Target) -> list[Tuple[np.ndarray, np.ndarray, float]]:
        """Extract segment data from target using episode reference information"""
        import os

        import numpy as np

        from rlhfblender.utils import process_env_name

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


class UnifiedFeedbackDataset(FeedbackDataset):
    """Wrapper that accepts Option 1 feedback objects"""

    def __init__(
        self,
        feedbacks: list[StandardizedFeedback],
        n_feedback: int,
        env_name: str = "",
        noise_level: float = 0.0,
        segment_len: int = 50,
        env=None,
        seed: int = 1234,
        **kwargs,
    ):
        # Convert Option 1 feedbacks to legacy format
        legacy_data = FeedbackDatasetAdapter.convert_to_legacy_format(feedbacks)

        # Determine the primary feedback type
        # You might want to handle mixed types differently
        if feedbacks:
            feedback_type_map = {
                "rating": "evaluative",
                "ranking": "comparative",
                "comparison": "comparative",
                "demonstration": "demonstrative",
                "correction": "corrective",
                "feature_selection": "descriptive",
            }
            primary_type = feedback_type_map.get(feedbacks[0].feedback_type, "evaluative")
        else:
            primary_type = "evaluative"

        super().__init__(
            feedback_data=legacy_data,
            feedback_type=primary_type,
            n_feedback=n_feedback,
            env_name=env_name,
            noise_level=noise_level,
            segment_len=segment_len,
            env=env,
            seed=seed,
            **kwargs,
        )

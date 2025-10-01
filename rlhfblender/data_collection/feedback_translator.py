# feedback_translator.py
from typing import Dict, List, Optional

from rlhfblender.data_collection.feedback_dataset_adapter import UnifiedFeedbackDataset
from rlhfblender.data_models.feedback_models import (
    ClusterRating,
    Correction,
    Demonstration,
    Entire,
    Episode,
    FeatureSelection,
    FeedbackType,
    MetaFeedback,
    Origin,
    Ranking,
    Rating,
    Segment,
    SimplifiedFeedbackType,
    StandardizedFeedback,
    State,
    Target,
    TextFeedback,
    UnprocessedFeedback,
)
from rlhfblender.data_models.global_models import Environment, Experiment
from rlhfblender.logger.logger import Logger


class FeedbackTranslator:
    """
    Translates UnprocessedFeedback into the simplified Feedback format.
    """

    def __init__(self, experiment: Experiment, env: Environment, logger: Logger = None):
        self.experiment = experiment
        self.env = env
        self.logger = logger
        self.feedback_id = 0
        self.feedback_buffer: List[StandardizedFeedback] = []

    def set_translator(self, experiment: Experiment, env: Environment, logger: Logger) -> str:
        """Sets the experiment and environment for the translator"""
        self.experiment = experiment
        self.env = env
        self.logger = logger
        self.reset()
        return "translator_set"

    def reset(self) -> None:
        """Resets the feedback translator"""
        self.feedback_id = 0
        self.feedback_buffer = []

    def _create_target(self, target_dict: dict, granularity: str) -> Optional[Target]:
        """Create appropriate Target object from dictionary"""
        if not target_dict:
            return None

        print("TARGET DICT:", target_dict)

        # Extract reference information (nested dict)
        reference = target_dict.get("reference", {})
        if isinstance(reference, dict):
            # New nested structure
            ref_kwargs = {
                "env_name": reference.get("env_name", ""),
                "benchmark_type": reference.get("benchmark_type", ""),
                "benchmark_id": reference.get("benchmark_id", -1),
                "checkpoint_step": reference.get("checkpoint_step", -1),
                "episode_num": reference.get("episode_num", -1),
            }
        else:
            # Fallback for old string format (if any)
            ref_kwargs = {
                "env_name": "",
                "benchmark_type": "",
                "benchmark_id": -1,
                "checkpoint_step": -1,
                "episode_num": -1,
            }

        base_kwargs = {
            "target_id": target_dict.get("target_id", ""),
            "origin": Origin[target_dict.get("origin", "offline")],
            "timestamp": target_dict.get("timestamp", -1),
            "data": target_dict.get("data", None),  # Store actual trajectory data
            **ref_kwargs,
        }

        if granularity == "episode":
            return Episode(**base_kwargs)
        elif granularity == "state":
            return State(**base_kwargs, step=target_dict.get("step", -1))
        elif granularity == "segment":
            return Segment(**base_kwargs, start=target_dict.get("start", -1), end=target_dict.get("end", -1))
        elif granularity == "entire":
            return Entire(**base_kwargs)

        return None

    def _map_feedback_type(self, old_type: FeedbackType) -> SimplifiedFeedbackType:
        """Map old feedback type to new simplified type"""
        mapping = {
            FeedbackType.rating: SimplifiedFeedbackType.rating,
            FeedbackType.ranking: SimplifiedFeedbackType.ranking,
            FeedbackType.comparison: SimplifiedFeedbackType.comparison,
            FeedbackType.correction: SimplifiedFeedbackType.correction,
            FeedbackType.demonstration: SimplifiedFeedbackType.demonstration,
            FeedbackType.clusterRating: SimplifiedFeedbackType.cluster_rating,
            FeedbackType.featureSelection: SimplifiedFeedbackType.feature_selection,
            FeedbackType.text: SimplifiedFeedbackType.text,
            FeedbackType.meta: SimplifiedFeedbackType.meta,
        }
        return mapping.get(old_type, SimplifiedFeedbackType.text)

    def give_feedback(self, session_id: str, feedback: UnprocessedFeedback) -> StandardizedFeedback:
        """Convert UnprocessedFeedback to simplified Feedback format"""

        # Base kwargs for all feedback types
        base_kwargs = {
            "feedback_id": self.feedback_id,
            "timestamp": feedback.timestamp,
            "session_id": session_id,
            "granularity": feedback.granularity,
        }

        # Map to new feedback type
        new_feedback_type = self._map_feedback_type(feedback.feedback_type)

        # Handle each feedback type
        if feedback.feedback_type == FeedbackType.rating:
            targets = [self._create_target(feedback.targets[0], feedback.granularity)] if feedback.targets else []
            content = Rating(score=feedback.score or 0.0)

        elif feedback.feedback_type in [FeedbackType.ranking, FeedbackType.comparison]:
            targets = [self._create_target(t, feedback.granularity) for t in feedback.targets]
            content = Ranking(preferences=feedback.preferences or [])

        elif feedback.feedback_type == FeedbackType.correction:
            targets = [self._create_target(t, feedback.granularity) for t in feedback.targets]
            content = Correction(action_preferences=feedback.action_preferences or [], path=feedback.correction_path)

        elif feedback.feedback_type == FeedbackType.demonstration:
            targets = [self._create_target(feedback.targets[0], feedback.granularity)] if feedback.targets else []
            # For demonstrations, the actions are stored in the target data
            content = Demonstration(actions=[], path=feedback.demonstration_path)

        elif feedback.feedback_type == FeedbackType.featureSelection:
            targets = [self._create_target(feedback.targets[0], feedback.granularity)] if feedback.targets else []
            content = FeatureSelection(features=feedback.feature_selection or [], importance=feedback.feature_importance)

        elif feedback.feedback_type == FeedbackType.clusterRating:
            targets = [self._create_target(t, feedback.granularity) for t in feedback.targets]
            content = ClusterRating(cluster_label=feedback.cluster_label or "Unknown", score=feedback.score or 0.0)

        elif feedback.feedback_type == FeedbackType.text:
            targets = [self._create_target(feedback.targets[0], feedback.granularity)] if feedback.targets else []
            content = TextFeedback(text=feedback.text_feedback or "")

        elif feedback.feedback_type == FeedbackType.meta:
            targets = []  # Meta feedback doesn't need targets
            content = MetaFeedback(action=feedback.meta_action or "")

        else:
            # Default to text feedback
            targets = [self._create_target(feedback.targets[0], feedback.granularity)] if feedback.targets else []
            content = TextFeedback(text=feedback.text_feedback or "")

        # Create the unified feedback object
        unified_feedback = StandardizedFeedback(
            **base_kwargs,
            feedback_type=new_feedback_type,
            targets=[t for t in targets if t is not None],  # Filter out None targets
            content=content
        )

        self.feedback_id += 1

        # Log the original feedback
        if self.logger:
            self.logger.log_raw(feedback)

        # Add to buffer
        self.feedback_buffer.append(unified_feedback)

        return unified_feedback

    def process(self) -> List[StandardizedFeedback]:
        """
        Process and deduplicate feedback buffer.
        Returns the processed feedback list.
        """
        # De-duplicate feedback in the buffer
        feedback_dict: Dict[tuple, StandardizedFeedback] = {}

        for feedback in self.feedback_buffer:
            # Skip meta feedback from deduplication
            if feedback.feedback_type == SimplifiedFeedbackType.meta:
                if self.logger:
                    self.logger.log(feedback)
                continue

            # Create deduplication key
            if feedback.targets:
                # Use first target's ID and feedback type as key
                key = (feedback.targets[0].target_id, feedback.feedback_type)
                feedback_dict[key] = feedback
            else:
                # No targets, use timestamp and type
                key = (feedback.timestamp, feedback.feedback_type)
                feedback_dict[key] = feedback

        # Get deduplicated feedback
        processed_feedback = list(feedback_dict.values())

        # Log all processed feedback
        if self.logger:
            for feedback in processed_feedback:
                self.logger.log(feedback)

        # Clear buffer and return processed feedback
        self.feedback_buffer = []

        return processed_feedback

    def get_unified_dataset(self, n_feedback: int = -1, **dataset_kwargs) -> "UnifiedFeedbackDataset":
        """
        Create a UnifiedFeedbackDataset from the current feedback buffer.

        Args:
            n_feedback: Number of feedback samples to use (-1 for all)
            **dataset_kwargs: Additional arguments for the dataset

        Returns:
            UnifiedFeedbackDataset ready for training
        """
        # Process the buffer to get final feedback list
        processed_feedback = self.process()

        return UnifiedFeedbackDataset(
            feedbacks=processed_feedback, n_feedback=n_feedback, env_name=str(self.env) if self.env else "", **dataset_kwargs
        )

    @staticmethod
    def create_unified_dataset_from_processed(
        processed_feedback: list["StandardizedFeedback"], n_feedback: int = -1, env_name: str = "", **dataset_kwargs
    ) -> "UnifiedFeedbackDataset":
        """
        Create a UnifiedFeedbackDataset from already processed feedback.

        Args:
            processed_feedback: List of StandardizedFeedback objects
            n_feedback: Number of feedback samples to use (-1 for all)
            env_name: Environment name
            **dataset_kwargs: Additional arguments for the dataset

        Returns:
            UnifiedFeedbackDataset ready for training
        """
        return UnifiedFeedbackDataset(feedbacks=processed_feedback, n_feedback=n_feedback, env_name=env_name, **dataset_kwargs)

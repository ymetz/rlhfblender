"""
This module translates incoming feedback of different types into a common format.
"""

from rlhfblender.data_models.feedback_models import (
    AbsoluteFeedback,
    Actuality,
    Content,
    Description,
    Evaluation,
    FeedbackType,
    Granularity,
    Instruction,
    Intention,
    Relation,
    RelativeDescription,
    RelativeEvaluation,
    RelativeFeedback,
    RelativeInstruction,
    StandardizedFeedback,
    StandardizedFeedbackType,
    Text,
    UnprocessedFeedback,
    get_granularity,
    get_target,
)
from rlhfblender.data_models.global_models import Environment, Experiment
from rlhfblender.logger.logger import Logger


class FeedbackTranslator:
    """
    This class translates incoming feedback of different types into a common format (StandardizedFeedback).

    : param experiment: The experiment object
    : param env: The environment object
    """

    def __init__(self, experiment: Experiment, env: Environment, logger: Logger = None):
        self.experiment = experiment
        self.env = env

        self.feedback_id = 0

        self.logger = logger
        self.feedback_buffer = []

    def set_translator(self, experiment: Experiment, env: Environment, logger: Logger) -> str:
        """
        Sets the experiment and environment for the translator
        :param experiment: The experiment object
        :param env: The environment object
        :return: The logger ID
        """
        self.experiment = experiment
        self.env = env
        self.logger = logger
        self.reset()

    def reset(self) -> None:
        """
        Resets the feedback translator
        :return:
        """
        self.feedback_id = 0
        self.feedback_buffer = []

    def give_feedback(self, session_id: str, feedback: UnprocessedFeedback) -> StandardizedFeedback:
        """
        We get either a single number or a list of numbers as feedback. We need to translate this into a common format
        called StandardizedFeedback
        :param session_id: The session ID
        :param feedback: (UnprocessedFeedback) The feedback
        :return: (StandardizedFeedback) The standardized feedback
        """
        return_feedback = None

        if feedback.feedback_type == FeedbackType.rating:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.evaluate,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=get_granularity(feedback.granularity),
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Evaluation(score=feedback.score),
            )
        elif feedback.feedback_type == FeedbackType.ranking:
            return_feedback = RelativeFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.evaluate,
                    actuality=Actuality.observed,
                    relation=Relation.relative,
                    content=Content.instance,
                    granularity=Granularity.episode,
                ),
                target=[get_target(target, feedback.granularity) for target in feedback.targets],  # is a list in this case
                content=RelativeEvaluation(preferences=feedback.preferences),
            )
        elif feedback.feedback_type == FeedbackType.correction:
            return_feedback = RelativeFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.instruct,
                    actuality=Actuality.observed,
                    relation=Relation.relative,
                    content=Content.instance,
                    granularity=Granularity.state,
                ),
                target=[get_target(target, feedback.granularity) for target in feedback.targets],  # is a list in this case
                content=RelativeInstruction(action_preferences=feedback.action_preferences),
            )
        elif feedback.feedback_type == FeedbackType.demonstration:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.instruct,
                    actuality=Actuality.hypothetical,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=Granularity.state,
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Instruction(action=[]),  # Content is already in the target (i.e. states and actions)
            )
        elif feedback.feedback_type == FeedbackType.featureSelection:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.describe,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=Granularity.entire,
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Description(feature_selection=feedback.feature_selection),
            )
        elif feedback.feedback_type == FeedbackType.descriptivePreferences:
            return_feedback = RelativeFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.describe,
                    actuality=Actuality.observed,
                    relation=Relation.relative,
                    content=Content.instance,
                    granularity=Granularity.entire,
                ),
                target=[get_target(target, feedback.granularity) for target in feedback.targets],  # is a list in this case
                content=RelativeDescription(
                    feature_selections_preferences=feedback.feature_selections_preferences,
                    feature_importance_preferences=feedback.feature_importance_preferences,
                ),
            )
        elif feedback.feedback_type == FeedbackType.text:
            # More comprenhesive logic will follow
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.describe,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=Granularity.entire,
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Text(text=feedback.text_feedback),
            )
        elif feedback.feedback_type == FeedbackType.meta:
            # Meta Actions such as submit, skip, etc. can also be interpreted as (implicit) feedback
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.none,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.meta,
                    granularity=Granularity.entire,
                ),
            )

        self.feedback_id += 1

        self.logger.log_raw(feedback)

        self.feedback_buffer.append(return_feedback)

    def process(self, session_id: str) -> None:
        """
        Submits the content of the current feedback buffer to the feedback dataset
        :param session_id: The session ID
        :return: None
        """
        # De-duplicate feedback in the feedback buffer
        # If feedback.episode_id and feedback.feedback_type are the same, we can assume that the feedback is the same,
        # we just want to keep the latest one
        feedback_dict = {}
        for feedback in self.feedback_buffer:
            if isinstance(feedback, AbsoluteFeedback):
                feedback_dict[(feedback.target.target_id, feedback.feedback_type)] = feedback
            else:
                feedback_dict[(feedback.target[0].target_id, feedback.feedback_type)] = feedback

        self.feedback_buffer = list(feedback_dict.values())

        for feedback in self.feedback_buffer:
            self.logger.log(feedback)

        self.feedback_buffer = []

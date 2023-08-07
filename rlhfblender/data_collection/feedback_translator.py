"""
This module translates incoming feedback of different types into a common format.
"""

import numpy as np
from data_models.feedback_models import (AbsoluteFeedback, Actuality, Content,
                                         EpisodeFeedback, FeedbackType,
                                         Granularity, Intention, Relation,
                                         RelativeFeedback,
                                         StandardizedFeedback,
                                         StandardizedFeedbackType)
from data_models.global_models import Environment, Experiment
from logger import CSVLogger, JSONLogger, SQLLogger


class FeedbackTranslator:
    def __init__(self, experiment: Experiment, env: Environment):
        self.experiment = experiment
        self.env = env

        self.feedback_id = 0

        self.logger = (
            JSONLogger(experiment, env, "feedback")
            if experiment is not None and env is not None
            else None
        )
        self.feedback_buffer = []

    def set_translator(self, experiment: Experiment, env: Environment):
        self.experiment = experiment
        self.env = env

        self.logger = CSVLogger(experiment, env, "feedback")

        self.reset()

        return self.logger.logger_id

    def reset(self):
        self.feedback_id = 0
        self.logger.reset()
        self.feedback_buffer = []

    def give_feedback(self, feedback: EpisodeFeedback) -> StandardizedFeedback:
        """
        We get either a single number or a list of numbers as feedback. We need to translate this into a common format
        called StandardizedFeedback
        :param feedback:
        :return:
        """
        return_feedback = None
        if feedback.feedback_type == FeedbackType.rating:
            # Translation can happen directly
            if feedback.array_feedback:
                return_feedback = AbsoluteFeedback(
                    feedback_id=self.feedback_id,
                    feedback_timestamp=feedback.timestamp,
                    feedback_type=StandardizedFeedbackType(
                        intention=Intention.evaluate,
                        actuality=Actuality.observed,
                        relation=Relation.absolute,
                        content=Content.instance,
                        granularity=Granularity.segment,
                    ),
                    episode_id=feedback.episode_id,
                    rating=feedback.array_feedback,
                )
            else:
                return_feedback = AbsoluteFeedback(
                    feedback_id=self.feedback_id,
                    feedback_timestamp=feedback.timestamp,
                    feedback_type=StandardizedFeedbackType(
                        intention=Intention.evaluate,
                        actuality=Actuality.observed,
                        relation=Relation.absolute,
                        content=Content.instance,
                        granularity=Granularity.episode,
                    ),
                    episode_id=feedback.episode_id,
                    rating=[feedback.numeric_feedback]
                    if not feedback.numeric_assignment
                    else [
                        (feedback.numeric_feedback * nav)
                        for nav in feedback.numeric_assignment
                    ],
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
                episode_id=feedback.episode_id,
                rating=0.0,
            )

        self.feedback_id += 1

        self.logger.log(return_feedback)
        self.logger.log_raw(feedback)

        self.feedback_buffer.append(return_feedback)

    def submit(self, session_id: str):
        """
        Submits the content of the current feedback buffer to the feedback dataset
        :param session_id:
        :return:
        """

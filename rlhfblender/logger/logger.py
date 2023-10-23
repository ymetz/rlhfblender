"""
Logging component for feedback logging
"""
from abc import abstractmethod
from datetime import datetime
from typing import List

from data_models.feedback_models import (UnprocessedFeedback, FeedbackType,
                                         StandardizedFeedback)
from data_models.global_models import Environment, Experiment


class Logger:
    def __init__(self, exp: Experiment, env: Environment, suffix: str):
        self.exp = exp
        self.env = env
        self.suffix = suffix

        self.logger_id = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + exp.exp_name
            + "_"
            + suffix
        )
        # Replace all spaces with underscores
        self.logger_id = (
            self.logger_id.replace(" ", "_").replace(":", "-").replace("/", "-")
        )

    def reset(self):
        self.logger_id = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + self.exp.exp_name
            + "_"
            + self.env.env_name
            + "_"
            + self.suffix
        )

    @abstractmethod
    def log(self, feedback: StandardizedFeedback):
        """
        Logs the feedback to a file or database
        :param feedback: A StandardizedFeedback instance
        :return:
        """

    @abstractmethod
    def log_raw(self, feedback: UnprocessedFeedback):
        """
        Logs the feedback to a file or database
        :param feedback:
        :return:
        """

    @abstractmethod
    def read(self) -> List[StandardizedFeedback]:
        """
        Reads all feedback from the logger
        :return:
        """

    @abstractmethod
    def read_raw(self) -> List[UnprocessedFeedback]:
        """
        Reads all feedback from the logger
        :return:
        """

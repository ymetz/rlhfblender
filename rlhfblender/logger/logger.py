"""
Logging component for feedback logging
"""

from abc import abstractmethod
from datetime import datetime

from rlhfblender.data_models.feedback_models import (
    StandardizedFeedback,
    UnprocessedFeedback,
)
from rlhfblender.data_models.global_models import Environment, Experiment


class Logger:
    """
    This class implements a logger that logs feedback to a file or database.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp: Experiment, env: Environment, suffix: str):
        self.exp = exp
        self.env = env
        self.suffix = suffix

        self.logger_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + exp.exp_name + "_" + suffix
        # Replace all spaces with underscores
        self.logger_id = self.logger_id.replace(" ", "_").replace(":", "-").replace("/", "-")

    def reset(self) -> None:
        """
        Resets the logger
        :return: None
        """
        self.logger_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + self.exp.exp_name + "_" + self.suffix
        # Replace all spaces with underscores
        self.logger_id = self.logger_id.replace(" ", "_").replace(":", "-").replace("/", "-")

    @abstractmethod
    def log(self, feedback: StandardizedFeedback):
        """
        Logs the feedback to a file or database
        :param feedback: A StandardizedFeedback instance
        :return: None
        """

    @abstractmethod
    def log_raw(self, feedback: UnprocessedFeedback):
        """
        Logs the feedback to a file or database
        :param feedback:
        :return: None
        """

    @abstractmethod
    def read(self) -> list[StandardizedFeedback]:
        """
        Reads all feedback from the logger
        :return: The processed feedback
        """

    @abstractmethod
    def read_raw(self) -> list[UnprocessedFeedback]:
        """
        Reads all feedback from the logger
        :return: The unprocessed feedback
        """

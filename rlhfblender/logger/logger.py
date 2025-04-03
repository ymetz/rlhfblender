"""
Logging component for feedback logging with async support
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Awaitable, Union

from rlhfblender.data_models.feedback_models import (
    StandardizedFeedback,
    UnprocessedFeedback,
)
from rlhfblender.data_models.global_models import Environment, Experiment


class Logger(ABC):
    """
    This class implements a base logger that logs feedback to a file or database.
    All derived loggers should implement the abstract methods.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp: Experiment, env: Environment, suffix: str):
        """
        Initialize the logger

        :param exp: The experiment object
        :param env: The environment object
        :param suffix: The suffix for the logger ID
        """
        self.exp = exp
        self.env = env
        self.suffix = suffix
        self.logger_id = ""

        # For async initialization
        self.init_complete = asyncio.Event()

    def _generate_logger_id(self) -> None:
        """
        Generate a unique logger ID based on timestamp and experiment name
        """
        self.logger_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + self.exp.exp_name + "_" + self.suffix
        # Replace all spaces with underscores and sanitize the ID
        self.logger_id = self.logger_id.replace(" ", "_").replace(":", "-").replace("/", "-")

    async def reset(self, exp: Experiment, env: Environment, suffix: str = None) -> str:
        """
        Resets the logger asynchronously

        :param exp: The experiment object
        :param env: The environment object
        :param suffix: Optional suffix for the logger ID
        :return: The new logger ID
        """
        self.exp = exp
        self.env = env
        if suffix is not None:
            self.suffix = suffix

        self._generate_logger_id()
        return self.logger_id

    @abstractmethod
    def log(self, feedback: StandardizedFeedback) -> None:
        """
        Logs the standardized feedback

        :param feedback: A StandardizedFeedback instance
        :return: None
        """
        pass

    @abstractmethod
    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs the raw feedback

        :param feedback: An UnprocessedFeedback instance
        :return: None
        """
        pass

    @abstractmethod
    async def dump(self) -> None:
        """
        Dumps the stored feedback to the storage medium

        :return: None
        """
        pass

    @abstractmethod
    async def dump_raw(self) -> None:
        """
        Dumps the stored raw feedback to the storage medium

        :return: None
        """
        pass

    @abstractmethod
    def read(self) -> Union[list[StandardizedFeedback], Awaitable[list[StandardizedFeedback]]]:
        """
        Reads all feedback from the logger

        :return: The processed feedback or a coroutine that resolves to the feedback
        """
        pass

    @abstractmethod
    def read_raw(self) -> Union[list[UnprocessedFeedback], Awaitable[list[UnprocessedFeedback]]]:
        """
        Reads all feedback from the logger

        :return: The unprocessed feedback or a coroutine that resolves to the feedback
        """
        pass

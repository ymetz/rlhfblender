import asyncio
import json

from rlhfblender.data_models.feedback_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()


class JSONLogger(Logger):
    """
    This class implements a logger that logs feedback to a json file.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp, env, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []

        self.logger_json_path = "logs/" + self.logger_id + ".json"
        self.raw_logger_json_path = "logs/" + self.logger_id + "_raw.json"

        self.init_empty_json()

    def init_empty_json(self) -> None:
        """
        Initializes the json file with empty lists
        :return: None
        """
        # Initialize the json file with empty lists
        with open(self.logger_json_path, "w") as f:
            f.write(json.dumps([]))

        with open(self.raw_logger_json_path, "w") as f:
            f.write(json.dumps([]))

    def reset(self, exp: Experiment, env: Environment, suffix: str = None) -> str:
        """
        Resets the logger
        :return: None
        """
        super().reset(exp, env, suffix)
        self.logger_json_path = "logs/" + self.logger_id + ".json"
        self.raw_logger_json_path = "logs/" + self.logger_id + "_raw.json"

        self.init_empty_json()

        return self.logger_id

    def log(self, feedback: StandardizedFeedback) -> None:
        """
        Logs standardized feedback
        :param feedback: The feedback
        :return: None
        """
        self.feedback.append(feedback)
        _task = asyncio.create_task(self.dump())
        background_tasks.add(_task)

    def read(self) -> list[StandardizedFeedback]:
        """
        Reads the feedback from the logger
        :return: The feedback
        """
        return self.feedback

    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs raw feedback
        :param feedback: The feedback
        :return: None
        """
        self.raw_feedback.append(feedback)
        _task = asyncio.create_task(self.dump_raw())
        background_tasks.add(_task)

    def read_raw(self) -> list[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger
        :return: The raw feedback
        """
        return self.raw_feedback

    async def dump(self) -> None:
        """
        Dumps the processed feedback to the json file
        :return: None
        """
        # Append the feedback to the list in the json file
        with open(self.logger_json_path, "a") as f:
            for feedback in self.feedback:
                f.write(json.dumps(feedback.json()) + "\n")
        self.feedback = []

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the json file
        :return: None
        """
        # Append the feedback to the json file
        with open(self.raw_logger_json_path, "a") as f:
            for feedback in self.raw_feedback:
                f.write(json.dumps(feedback.json()) + "\n")
        self.raw_feedback = []

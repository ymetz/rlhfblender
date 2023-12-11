import asyncio
import csv
import os
from typing import List

from pydantic import BaseModel

from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback

from .logger import Logger


class CSVLogger(Logger):
    """
    This class implements a logger that logs feedback to a csv file.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp, env, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback: List[UnprocessedFeedback] = []
        self.feedback: List[StandardizedFeedback] = []

        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

    def reset(self) -> None:
        """
        Resets the logger
        :return: None
        """
        super().reset()
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

    def log(self, feedback):
        """
        Logs standardized feedback
        :param feedback: The feedback
        :return: None
        """
        self.feedback.append(feedback)
        _task = asyncio.create_task(self.dump())
        

    def read(self) -> List[StandardizedFeedback]:
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

    def read_raw(self) -> List[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger
        :return: The raw feedback
        """
        return self.raw_feedback

    async def dump(self) -> None:
        """
        Dumps the processed feedback to the csv file
        :return: None
        """
        # Append the feedback to the list in the csv file: Translate feedback to csv format
        if len(self.feedback) > 0:
            fb: BaseModel = self.feedback[0]
            with open(self.logger_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                if os.path.getsize(self.logger_csv_path) == 0:
                    writer.writeheader()
                for feedback in self.feedback:
                    writer.writerow(feedback.dict())
        self.feedback = []

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the csv file
        :return: None
        """
        if len(self.raw_feedback) > 0:
            fb = self.raw_feedback[0]
            # Create dir if not exists
            with open(self.raw_logger_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                if os.path.getsize(self.raw_logger_csv_path) == 0:
                    writer.writeheader()
                for feedback in self.raw_feedback:
                    writer.writerow(feedback.dict())

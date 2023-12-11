import asyncio
from typing import List

from rlhfblender.data_handling import database_handler
from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback

from .logger import Logger


class SQLLogger(Logger):
    """
    This class implements a logger that logs feedback to a sql database.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp, env, suffix, db):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []

        self.sql_table = self.logger_id
        self.db = db

    def reset(self):
        """
        Resets the logger
        :return: None
        """
        super().reset()

        self.sql_table = self.logger_id

        database_handler.create_table_from_model(self.db, StandardizedFeedback, self.sql_table)
        database_handler.create_table_from_model(self.db, UnprocessedFeedback, self.sql_table + "_raw")

    def log(self, feedback: StandardizedFeedback) -> None:
        """
        Logs standardized feedback
        :param feedback: The feedback
        :return: None
        """
        self.feedback.append(feedback)
        _task = asyncio.create_task(self.dump())

    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs raw feedback
        :param feedback: The feedback
        :return: None
        """
        self.raw_feedback.append(feedback)
        _task = asyncio.create_task(self.dump_raw())

    async def dump(self) -> None:
        """
        Writes the feedback to the database
        :return: None
        """
        # Write the feedback to the database
        for feedback in self.feedback:
            await database_handler.add_entry(self.db, StandardizedFeedback, feedback)
        self.feedback = []

    async def dump_raw(self) -> None:
        """
        Writes the raw feedback to the database
        :return: None
        """
        # Append the feedback to the json file
        for feedback in self.raw_feedback:
            await database_handler.add_entry(self.db, UnprocessedFeedback, feedback)
        self.raw_feedback = []

    async def read(self) -> List[StandardizedFeedback]:
        """
        Reads the processed feedback from the logger
        :return: The feedback
        """
        return await database_handler.get_all(self.db, StandardizedFeedback, self.sql_table)

    async def read_raw(self) -> List[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger
        :return: The raw feedback
        """
        return await database_handler.get_all(self.db, UnprocessedFeedback, self.sql_table + "_raw")

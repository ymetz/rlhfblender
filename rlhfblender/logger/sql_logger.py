import asyncio
from typing import List

from rlhfblender.data_handling import database_handler
from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.logger import Logger


class SQLLogger(Logger):
    def __init__(self, exp, env, suffix, db):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []

        self.sql_table = self.logger_id
        self.db = db

    def reset(self):
        super().reset()

        self.sql_table = self.logger_id

        database_handler.create_table_from_model(self.db, StandardizedFeedback, self.sql_table)
        database_handler.create_table_from_model(self.db, UnprocessedFeedback, self.sql_table + "_raw")

    def log(self, feedback):
        self.feedback.append(feedback)
        asyncio.create_task(self.dump())

    def log_raw(self, feedback):
        self.raw_feedback.append(feedback)
        asyncio.create_task(self.dump_raw())

    async def dump(self):
        # Write the feedback to the database
        for feedback in self.feedback:
            await database_handler.add_entry(self.db, StandardizedFeedback, feedback)
        self.feedback = []

    async def dump_raw(self):
        # Append the feedback to the json file
        for feedback in self.raw_feedback:
            await database_handler.add_entry(self.db, UnprocessedFeedback, feedback)
        self.raw_feedback = []

    async def read(self) -> List[StandardizedFeedback]:
        return await database_handler.get_all(self.db, StandardizedFeedback, self.sql_table)

    async def read_raw(self) -> List[UnprocessedFeedback]:
        return await database_handler.get_all(self.db, UnprocessedFeedback, self.sql_table + "_raw")

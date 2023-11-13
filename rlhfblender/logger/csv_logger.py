import asyncio
import csv
import os

from pydantic import BaseModel

from rlhfblender.logger.logger import Logger


class CSVLogger(Logger):
    def __init__(self, exp, env, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []

        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

    def reset(self):
        super().reset()
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

    def log(self, feedback):
        self.feedback.append(feedback)
        asyncio.create_task(self.dump())

    def read(self):
        return self.feedback

    def log_raw(self, feedback):
        self.raw_feedback.append(feedback)
        asyncio.create_task(self.dump_raw())

    def read_raw(self):
        return self.raw_feedback

    async def dump(self):
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

    async def dump_raw(self):
        if len(self.raw_feedback) > 0:
            fb = self.raw_feedback[0]
            # Create dir if not exists
            with open(self.raw_logger_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                if os.path.getsize(self.raw_logger_csv_path) == 0:
                    writer.writeheader()
                for feedback in self.raw_feedback:
                    writer.writerow(feedback.dict())

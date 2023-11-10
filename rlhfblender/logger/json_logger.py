import asyncio
import json

from rlhfblender.logger.logger import Logger


class JSONLogger(Logger):
    def __init__(self, exp, env, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []

        self.logger_json_path = "logs/" + self.logger_id + ".json"
        self.raw_logger_json_path = "logs/" + self.logger_id + "_raw.json"

        self.init_empty_json()

    def init_empty_json(self):
        # Initialize the json file with empty lists
        with open(self.logger_json_path, "w") as f:
            f.write(json.dumps([]))

        with open(self.raw_logger_json_path, "w") as f:
            f.write(json.dumps([]))

    def reset(self):
        super().reset()
        self.logger_json_path = "logs/" + self.logger_id + ".json"
        self.raw_logger_json_path = "logs/" + self.logger_id + "_raw.json"

        self.init_empty_json()

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
        # Append the feedback to the list in the json file
        with open(self.logger_json_path, "a") as f:
            for feedback in self.feedback:
                f.write(json.dumps(feedback.json()) + "\n")
        self.feedback = []

    async def dump_raw(self):
        # Append the feedback to the json file
        with open(self.raw_logger_json_path, "a") as f:
            for feedback in self.raw_feedback:
                f.write(json.dumps(feedback.json()) + "\n")
        self.raw_feedback = []

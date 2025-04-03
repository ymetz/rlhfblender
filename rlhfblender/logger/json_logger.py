import asyncio
import json
import os

from rlhfblender.data_models.feedback_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()


class JSONLogger(Logger):
    """
    This class implements a logger that logs feedback to a json file asynchronously.

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

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_empty_json_async())
        background_tasks.add(self._init_task)

        # Flag to track initialization completion
        self.init_complete = asyncio.Event()

    async def _init_empty_json_async(self) -> None:
        """
        Initializes the json files with empty lists asynchronously

        :return: None
        """
        try:
            # Use run_in_executor for file I/O operations to avoid blocking
            loop = asyncio.get_event_loop()

            async def write_empty_json(file_path):
                def _write():
                    with open(file_path, "w") as f:
                        f.write(json.dumps([]))

                await loop.run_in_executor(None, _write)

            # Initialize both files concurrently
            await asyncio.gather(write_empty_json(self.logger_json_path), write_empty_json(self.raw_logger_json_path))

            # Mark initialization as complete
            self.init_complete.set()

        except Exception as e:
            print(f"Error in async JSON initialization: {e}")
            # Set the event anyway to avoid hanging
            self.init_complete.set()

    async def reset(self, exp: Experiment, env: Environment, suffix: str = None) -> str:
        """
        Resets the logger asynchronously

        :return: The logger ID
        """
        # Cancel the previous init task if it's still running
        if hasattr(self, "_init_task") and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass

        # Reset base logger
        await super().reset(exp, env, suffix)

        # Reset paths and state
        self.logger_json_path = "logs/" + self.logger_id + ".json"
        self.raw_logger_json_path = "logs/" + self.logger_id + "_raw.json"
        self.feedback = []
        self.raw_feedback = []
        self.init_complete = asyncio.Event()

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_empty_json_async())
        background_tasks.add(self._init_task)

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
        Dumps the processed feedback to the json file asynchronously

        :return: None
        """
        if not self.feedback:
            return

        # Wait for initialization to complete
        await self.init_complete.wait()

        feedback_to_dump = self.feedback.copy()
        self.feedback = []

        try:
            # Use run_in_executor for file I/O operations
            loop = asyncio.get_event_loop()

            def _write():
                with open(self.logger_json_path, "a") as f:
                    for feedback in feedback_to_dump:
                        f.write(json.dumps(feedback.model_dump()) + "\n")

            await loop.run_in_executor(None, _write)

        except Exception as e:
            print(f"Error dumping JSON feedback: {e}")
            # Add back to list if write fails
            self.feedback.extend(feedback_to_dump)

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the json file asynchronously

        :return: None
        """
        if not self.raw_feedback:
            return

        # Wait for initialization to complete
        await self.init_complete.wait()

        raw_feedback_to_dump = self.raw_feedback.copy()
        self.raw_feedback = []

        try:
            # Use run_in_executor for file I/O operations
            loop = asyncio.get_event_loop()

            def _write():
                with open(self.raw_logger_json_path, "a") as f:
                    for feedback in raw_feedback_to_dump:
                        f.write(json.dumps(feedback.dict()) + "\n")

            await loop.run_in_executor(None, _write)

        except Exception as e:
            print(f"Error dumping raw JSON feedback: {e}")
            # Add back to list if write fails
            self.raw_feedback.extend(raw_feedback_to_dump)

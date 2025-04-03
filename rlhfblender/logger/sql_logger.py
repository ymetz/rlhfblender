import asyncio

from rlhfblender.data_handling import database_handler
from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()


class SQLLogger(Logger):
    """
    This class implements a logger that logs feedback to a sql database asynchronously.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    :param db: The database connection
    """

    def __init__(self, exp, env, suffix, db):
        super().__init__(exp, env, suffix)
        self.raw_feedback = []
        self.feedback = []
        self.sql_table = self.logger_id
        self.db = db

        # Initialization flag
        self.init_complete = asyncio.Event()

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_tables_async())
        background_tasks.add(self._init_task)

    async def _init_tables_async(self) -> None:
        """
        Initialize SQL tables asynchronously

        :return: None
        """
        try:
            # Create tables for standardized and raw feedback
            await database_handler.create_table_from_model(self.db, StandardizedFeedback, self.sql_table)
            await database_handler.create_table_from_model(self.db, UnprocessedFeedback, self.sql_table + "_raw")

            # Mark initialization as complete
            self.init_complete.set()

        except Exception as e:
            print(f"Error initializing SQL tables: {e}")
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

        # Reset the base logger
        await super().reset(exp, env, suffix)

        # Reset state
        self.sql_table = self.logger_id
        self.feedback = []
        self.raw_feedback = []
        self.init_complete = asyncio.Event()

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_tables_async())
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

    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs raw feedback

        :param feedback: The feedback
        :return: None
        """
        self.raw_feedback.append(feedback)
        _task = asyncio.create_task(self.dump_raw())
        background_tasks.add(_task)

    async def dump(self) -> None:
        """
        Writes the feedback to the database asynchronously

        :return: None
        """
        if not self.feedback:
            return

        # Wait for initialization to complete
        await self.init_complete.wait()

        # Make a copy of the feedback items to process
        feedback_to_dump = self.feedback.copy()
        self.feedback = []

        try:
            # Process each feedback item
            for feedback in feedback_to_dump:
                try:
                    await database_handler.add_entry(self.db, StandardizedFeedback, feedback)
                except Exception as e:
                    print(f"Error adding feedback entry to database: {e}")
                    # Add back to list if adding fails
                    self.feedback.append(feedback)

        except Exception as e:
            print(f"Error in dump operation: {e}")
            # Keep feedback in memory if operation fails
            self.feedback.extend(feedback_to_dump)

    async def dump_raw(self) -> None:
        """
        Writes the raw feedback to the database asynchronously

        :return: None
        """
        if not self.raw_feedback:
            return

        # Wait for initialization to complete
        await self.init_complete.wait()

        # Make a copy of the raw feedback items to process
        raw_feedback_to_dump = self.raw_feedback.copy()
        self.raw_feedback = []

        try:
            # Process each raw feedback item
            for feedback in raw_feedback_to_dump:
                try:
                    await database_handler.add_entry(self.db, UnprocessedFeedback, feedback)
                except Exception as e:
                    print(f"Error adding raw feedback entry to database: {e}")
                    # Add back to list if adding fails
                    self.raw_feedback.append(feedback)

        except Exception as e:
            print(f"Error in dump_raw operation: {e}")
            # Keep feedback in memory if operation fails
            self.raw_feedback.extend(raw_feedback_to_dump)

    async def read(self) -> list[StandardizedFeedback]:
        """
        Reads the processed feedback from the logger

        :return: The feedback
        """
        # Wait for initialization to complete
        await self.init_complete.wait()

        try:
            # Get all entries from the database plus any pending in-memory entries
            db_entries = await database_handler.get_all(self.db, StandardizedFeedback, self.sql_table)
            return list(db_entries) + self.feedback
        except Exception as e:
            print(f"Error reading feedback from database: {e}")
            # Return only in-memory entries if database read fails
            return self.feedback

    async def read_raw(self) -> list[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger

        :return: The raw feedback
        """
        # Wait for initialization to complete
        await self.init_complete.wait()

        try:
            # Get all raw entries from the database plus any pending in-memory entries
            db_entries = await database_handler.get_all(self.db, UnprocessedFeedback, self.sql_table + "_raw")
            return list(db_entries) + self.raw_feedback
        except Exception as e:
            print(f"Error reading raw feedback from database: {e}")
            # Return only in-memory entries if database read fails
            return self.raw_feedback

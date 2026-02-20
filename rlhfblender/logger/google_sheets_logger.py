import asyncio
import os

import gspread
from gspread.exceptions import SpreadsheetNotFound

from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()


class GoogleSheetsLogger(Logger):
    """
    This class implements a logger that logs feedback to Google Sheets asynchronously.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    :param credentials_file: Path to the Google service account credentials JSON file
    :param spreadsheet_name: Name of the Google spreadsheet to use (will be created if it doesn't exist)
    """

    def __init__(
        self,
        exp: Experiment,
        env: Environment,
        suffix,
        credentials_file: str = "google-service-account.json",
        spreadsheet_name: str | None = None,
    ):
        super().__init__(exp, env, suffix)
        self.raw_feedback: list[UnprocessedFeedback] = []
        self.feedback: list[StandardizedFeedback] = []

        # Google Sheets specific configurations
        self.credentials_file = credentials_file
        self.spreadsheet_name = spreadsheet_name or f"Survey_Results_{self.logger_id}"

        # Initialization state tracking
        self.init_complete = asyncio.Event()
        self.client = None
        self.spreadsheet = None
        self.worksheet_name = "processed_feedback"
        self.raw_worksheet_name = "raw_feedback"

        # Caches for feedback received before initialization completes
        self.feedback_cache = []
        self.raw_feedback_cache = []

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_google_sheets_async())
        background_tasks.add(self._init_task)

    async def _init_google_sheets_async(self):
        """Initialize Google Sheets client asynchronously and create spreadsheet if needed"""
        try:
            # Use ThreadPoolExecutor for blocking gspread operations
            loop = asyncio.get_event_loop()

            def init_client():
                try:
                    client = gspread.service_account(self.credentials_file)

                    # Try to open existing spreadsheet or create a new one
                    try:
                        spreadsheet = client.open(self.spreadsheet_name)
                    except SpreadsheetNotFound:
                        print("Spreadsheet not found, creating a new one...", self.spreadsheet_name)
                        spreadsheet = client.create(self.spreadsheet_name)

                    # Create worksheets if they don't exist
                    worksheet_list = [ws.title for ws in spreadsheet.worksheets()]

                    if self.worksheet_name not in worksheet_list:
                        spreadsheet.add_worksheet(title=self.worksheet_name, rows=1000, cols=50)

                    if self.raw_worksheet_name not in worksheet_list:
                        spreadsheet.add_worksheet(title=self.raw_worksheet_name, rows=1000, cols=50)

                    # sharing the spreadsheet with the provided email
                    share_email = os.getenv("GOOGLE_SHEETS_SHARE_EMAIL")
                    if share_email:
                        spreadsheet.share(share_email, perm_type="user", role="writer")

                    return client, spreadsheet
                except Exception as e:
                    print(f"Error in thread initializing Google Sheets: {e}")
                    return None, None

            # Run the blocking operations in a thread
            self.client, self.spreadsheet = await loop.run_in_executor(None, init_client)

            # Process any cached feedback
            if self.client and self.spreadsheet:
                if self.feedback_cache:
                    for feedback_item in self.feedback_cache:
                        self.feedback.append(feedback_item)
                    self.feedback_cache = []
                    await self.dump()

                if self.raw_feedback_cache:
                    for raw_item in self.raw_feedback_cache:
                        self.raw_feedback.append(raw_item)
                    self.raw_feedback_cache = []
                    await self.dump_raw()

            # Set the initialization as complete
            self.init_complete.set()

        except Exception as e:
            print(f"Error in async Google Sheets initialization: {e}")
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
        self.spreadsheet_name = f"Survey_Results_{self.logger_id}"
        self.init_complete = asyncio.Event()
        self.client = None
        self.spreadsheet = None
        self.feedback = []
        self.raw_feedback = []
        self.feedback_cache = []
        self.raw_feedback_cache = []

        # Start async initialization
        self._init_task = asyncio.create_task(self._init_google_sheets_async())
        background_tasks.add(self._init_task)

        return self.logger_id

    def log(self, feedback: StandardizedFeedback):
        """
        Logs standardized feedback, caching if initialization is not complete

        :param feedback: The feedback
        :return: None
        """
        if not self.init_complete.is_set():
            # Cache the feedback until initialization completes
            self.feedback_cache.append(feedback)
        else:
            self.feedback.append(feedback)
            _task = asyncio.create_task(self.dump())
            background_tasks.add(_task)

    def read(self) -> list[StandardizedFeedback]:
        """
        Reads the feedback from the logger, including cached items

        :return: The feedback
        """
        return self.feedback + self.feedback_cache

    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs raw feedback, caching if initialization is not complete

        :param feedback: The feedback
        :return: None
        """
        if not self.init_complete.is_set():
            # Cache the raw feedback until initialization completes
            self.raw_feedback_cache.append(feedback)
        else:
            self.raw_feedback.append(feedback)
            _task = asyncio.create_task(self.dump_raw())
            background_tasks.add(_task)

    def read_raw(self) -> list[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger, including cached items

        :return: The raw feedback
        """
        return self.raw_feedback + self.raw_feedback_cache

    async def dump(self) -> None:
        """
        Dumps the processed feedback to Google Sheets

        :return: None
        """
        if not self.client or len(self.feedback) == 0:
            return

        try:
            # Use run_in_executor for the blocking gspread operations
            loop = asyncio.get_event_loop()

            def append_to_sheet():
                try:
                    # Get the right worksheet
                    worksheet = self.spreadsheet.worksheet(self.worksheet_name)

                    # Check if headers need to be added (empty sheet)
                    if worksheet.row_count <= 1:  # Only has header or is empty
                        fb_dict = self.feedback[0].model_dump()
                        headers = list(fb_dict.keys())
                        worksheet.append_row(headers)

                    # Convert feedback to rows and append to sheet
                    rows_to_append = []
                    for feedback in self.feedback:
                        fb_dict = feedback.model_dump()
                        # Convert all values to strings to avoid type issues
                        row = [str(fb_dict[key]) for key in fb_dict.keys()]
                        rows_to_append.append(row)

                    # Use batch append for better performance
                    if rows_to_append:
                        worksheet.append_rows(rows_to_append)

                    return True
                except Exception as e:
                    print(f"Error writing to Google Sheets in thread: {e}")
                    return False

            success = await loop.run_in_executor(None, append_to_sheet)

            if success:
                # Clear the feedback list after successful write
                self.feedback = []

        except Exception as e:
            print(f"Error in async dump to Google Sheets: {e}")
            # Keep feedback in memory if write fails

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to Google Sheets

        :return: None
        """
        if not self.client or len(self.raw_feedback) == 0:
            return

        try:
            # Use run_in_executor for the blocking gspread operations
            loop = asyncio.get_event_loop()

            def append_raw_to_sheet():
                try:
                    # Get the right worksheet
                    worksheet = self.spreadsheet.worksheet(self.raw_worksheet_name)

                    # Check if headers need to be added (empty sheet)
                    if worksheet.row_count <= 1:  # Only has header or is empty
                        fb_dict = self.raw_feedback[0].dict()
                        headers = list(fb_dict.keys())
                        worksheet.append_row(headers)

                    # Convert feedback to rows and append to sheet
                    rows_to_append = []
                    for feedback in self.raw_feedback:
                        fb_dict = feedback.dict()
                        # Convert all values to strings to avoid type issues
                        row = [str(fb_dict[key]) for key in fb_dict.keys()]
                        rows_to_append.append(row)

                    # Use batch append for better performance
                    if rows_to_append:
                        worksheet.append_rows(rows_to_append)

                    return True
                except Exception as e:
                    print(f"Error writing raw feedback to Google Sheets in thread: {e}")
                    return False

            success = await loop.run_in_executor(None, append_raw_to_sheet)

            if success:
                # Clear the raw feedback list after successful write
                self.raw_feedback = []

        except Exception as e:
            print(f"Error in async dump_raw to Google Sheets: {e}")
            # Keep feedback in memory if write fails

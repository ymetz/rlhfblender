import asyncio
import os
from typing import Optional

import gspread

from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()


class GoogleSheetsLogger(Logger):
    """
    This class implements a logger that logs feedback to Google Sheets.

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
        spreadsheet_name: Optional[str] = None,
    ):
        super().__init__(exp, env, suffix)
        self.raw_feedback: list[UnprocessedFeedback] = []
        self.feedback: list[StandardizedFeedback] = []

        # Google Sheets specific configurations
        self.credentials_file = credentials_file
        self.spreadsheet_name = spreadsheet_name or f"Survey_Results_{self.logger_id}"
        self.client = None
        self.worksheet_name = "processed_feedback"
        self.raw_worksheet_name = "raw_feedback"

        # Initialize Google Sheets client
        # self._init_google_sheets()

    def _init_google_sheets(self):
        """Initialize Google Sheets client and create spreadsheet if needed"""
        try:
            self.client = gspread.service_account(self.credentials_file)

            # Try to open existing spreadsheet or create a new one
            try:
                self.spreadsheet = self.client.open(self.spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                print("Spreadsheet not found, creating a new one...", self.spreadsheet_name)
                self.spreadsheet = self.client.create(self.spreadsheet_name)

            # Create worksheets if they don't exist
            worksheet_list = [ws.title for ws in self.spreadsheet.worksheets()]

            if self.worksheet_name not in worksheet_list:
                self.spreadsheet.add_worksheet(title=self.worksheet_name, rows=1000, cols=50)

            if self.raw_worksheet_name not in worksheet_list:
                self.spreadsheet.add_worksheet(title=self.raw_worksheet_name, rows=1000, cols=50)

            # sharing the spreadsheet with the provided email
            share_email = os.getenv("GOOGLE_SHEETS_SHARE_EMAIL")
            if share_email:
                self.spreadsheet.share(share_email, perm_type="user", role="writer")

        except Exception as e:
            print(f"Error initializing Google Sheets: {e}")
            # Fallback to local CSV if Google Sheets fails
            self.client = None

    def reset(self, exp: Experiment, env: Environment, suffix: str = None) -> str:
        """
        Resets the logger

        :return: The logger ID
        """
        super().reset(exp, env, suffix)
        self.spreadsheet_name = f"Survey_Results_{self.logger_id}"

        # Re-initialize Google Sheets with new name
        self._init_google_sheets()
        return self.logger_id

    def log(self, feedback: StandardizedFeedback):
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
        Dumps the processed feedback to Google Sheets

        :return: None
        """
        if not self.client or len(self.feedback) == 0:
            return

        try:
            # Get the right worksheet
            worksheet = self.spreadsheet.worksheet(self.worksheet_name)

            # Check if headers need to be added (empty sheet)
            if worksheet.row_count <= 1:  # Only has header or is empty
                fb_dict = self.feedback[0].model_dump()
                headers = list(fb_dict.keys())
                worksheet.append_row(headers)

            # Convert feedback to rows and append to sheet
            for feedback in self.feedback:
                fb_dict = feedback.model_dump()
                # Convert all values to strings to avoid type issues
                row = [str(fb_dict[key]) for key in fb_dict.keys()]
                worksheet.append_row(row)

            # Clear the feedback list after successful write
            self.feedback = []

        except Exception as e:
            print(f"Error writing to Google Sheets: {e}")
            # Keep feedback in memory if write fails

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to Google Sheets

        :return: None
        """
        if not self.client or len(self.raw_feedback) == 0:
            return

        try:
            # Get the right worksheet
            worksheet = self.spreadsheet.worksheet(self.raw_worksheet_name)

            # Check if headers need to be added (empty sheet)
            if worksheet.row_count <= 1:  # Only has header or is empty
                fb_dict = self.raw_feedback[0].dict()
                headers = list(fb_dict.keys())
                worksheet.append_row(headers)

            # Convert feedback to rows and append to sheet
            for feedback in self.raw_feedback:
                fb_dict = feedback.dict()
                # Convert all values to strings to avoid type issues
                row = [str(fb_dict[key]) for key in fb_dict.keys()]
                worksheet.append_row(row)

            # Clear the raw feedback list after successful write
            self.raw_feedback = []

        except Exception as e:
            print(f"Error writing raw feedback to Google Sheets: {e}")
            # Keep feedback in memory if write fails

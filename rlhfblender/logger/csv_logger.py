import asyncio
import csv
import os
from pydantic import BaseModel

from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback
from rlhfblender.data_models.global_models import Environment, Experiment

from .logger import Logger

background_tasks = set()

class CSVLogger(Logger):
    """
    This class implements a logger that logs feedback to a csv file asynchronously.
    
    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """
    
    def __init__(self, exp: Experiment, env: Environment, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback: list[UnprocessedFeedback] = []
        self.feedback: list[StandardizedFeedback] = []
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialization flag
        self.init_complete = asyncio.Event()
        self.init_complete.set()  # CSV logger doesn't need lengthy initialization
    
    async def reset(self, exp: Experiment, env: Environment, suffix: str = None) -> str:
        """
        Resets the logger asynchronously
        
        :return: The logger ID
        """
        # Reset base logger
        await super().reset(exp, env, suffix)
        
        # Reset paths and state
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"
        self.feedback = []
        self.raw_feedback = []
        
        # Create empty files
        loop = asyncio.get_event_loop()
        
        async def create_empty_files():
            def _create_files():
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.logger_csv_path), exist_ok=True)
                
                # Create empty CSV files (will be populated with headers on first write)
                open(self.logger_csv_path, 'w').close()
                open(self.raw_logger_csv_path, 'w').close()
            
            await loop.run_in_executor(None, _create_files)
        
        # Start the file creation in background
        self._init_task = asyncio.create_task(create_empty_files())
        background_tasks.add(self._init_task)
        
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
        Dumps the processed feedback to the csv file asynchronously
        
        :return: None
        """
        if not self.feedback:
            return
        
        # Make a copy of the feedback items to process
        feedback_to_dump = self.feedback.copy()
        self.feedback = []
        
        try:
            # Use run_in_executor for file I/O operations
            loop = asyncio.get_event_loop()
            
            def _write_csv():
                try:
                    # Get fields from the first feedback item
                    fb: BaseModel = feedback_to_dump[0]
                    field_names = fb.model_dump().keys() if hasattr(fb, 'model_dump') else fb.dict().keys()
                    
                    # Check if file exists and is empty (needs headers)
                    file_exists = os.path.exists(self.logger_csv_path)
                    needs_header = not file_exists or os.path.getsize(self.logger_csv_path) == 0
                    
                    with open(self.logger_csv_path, "a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=field_names)
                        
                        if needs_header:
                            writer.writeheader()
                        
                        for item in feedback_to_dump:
                            # Use model_dump() for Pydantic v2, fallback to dict() for v1
                            row_data = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
                            writer.writerow(row_data)
                    
                    return True
                except Exception as e:
                    print(f"Error writing to CSV: {e}")
                    return False
            
            success = await loop.run_in_executor(None, _write_csv)
            
            if not success:
                # Add back to list if write fails
                self.feedback.extend(feedback_to_dump)
        
        except Exception as e:
            print(f"Error in async dump to CSV: {e}")
            # Keep feedback in memory if operation fails
            self.feedback.extend(feedback_to_dump)
    
    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the csv file asynchronously
        
        :return: None
        """
        if not self.raw_feedback:
            return
        
        # Make a copy of the raw feedback items to process
        raw_feedback_to_dump = self.raw_feedback.copy()
        self.raw_feedback = []
        
        try:
            # Use run_in_executor for file I/O operations
            loop = asyncio.get_event_loop()
            
            def _write_raw_csv():
                try:
                    # Get fields from the first feedback item
                    fb = raw_feedback_to_dump[0]
                    field_names = fb.model_dump().keys() if hasattr(fb, 'model_dump') else fb.dict().keys()
                    
                    # Check if file exists and is empty (needs headers)
                    file_exists = os.path.exists(self.raw_logger_csv_path)
                    needs_header = not file_exists or os.path.getsize(self.raw_logger_csv_path) == 0
                    
                    with open(self.raw_logger_csv_path, "a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=field_names)
                        
                        if needs_header:
                            writer.writeheader()
                        
                        for item in raw_feedback_to_dump:
                            # Use model_dump() for Pydantic v2, fallback to dict() for v1
                            row_data = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
                            writer.writerow(row_data)
                    
                    return True
                except Exception as e:
                    print(f"Error writing raw feedback to CSV: {e}")
                    return False
            
            success = await loop.run_in_executor(None, _write_raw_csv)
            
            if not success:
                # Add back to list if write fails
                self.raw_feedback.extend(raw_feedback_to_dump)
        
        except Exception as e:
            print(f"Error in async dump_raw to CSV: {e}")
            # Keep feedback in memory if operation fails
            self.raw_feedback.extend(raw_feedback_to_dump)
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import wandb

# Import the Logger class from the Stable Baselines 3 logging module
# This would typically be: from stable_baselines3.common.logger import Logger, KVWriter
# For your code, adjust the import as needed
from stable_baselines3.common.logger import KVWriter, Logger
from wandb.sdk.lib import telemetry as wb_telemetry

logger = logging.getLogger(__name__)


class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Weights & Biases's format.

    This custom format maintains a global step counter and does not reset between
    training runs.

    :param project: W&B project name
    :param run_name: W&B run name
    :param global_step_offset: Starting global step value to continue from
    :param config: Additional config values to log
    :param run_id: Run ID to resume a previous run (optional)
    """

    def __init__(
        self,
        global_step_offset: int = 0,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ):

        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True

        self.global_step = global_step_offset
        self._is_closed = False

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """
        Write key-value pairs to W&B, filtering excluded keys.

        Uses the internal global step counter instead of the provided step parameter.

        :param key_values: Dictionary of key-value pairs to log
        :param key_excluded: Dictionary of keys to excluded formats
        :param step: Step parameter (ignored in favor of global_step)
        """
        assert (
            not self._is_closed
        ), "The WandbOutputFormat was closed, please re-create one."

        # Filter out excluded keys
        log_dict = {}

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "wandb" in excluded:
                continue

            # Handle different types of values
            if isinstance(value, (int, float, str, bool, np.number)):
                log_dict[key] = value
            elif isinstance(value, (np.ndarray, list)) and len(value) == 1:
                # Handle scalar arrays
                log_dict[key] = value[0]
            elif isinstance(value, dict):
                # For dictionaries, add each key-value pair individually
                for subkey, subvalue in value.items():
                    log_dict[f"{key}/{subkey}"] = subvalue
            elif hasattr(value, "dtype"):
                # For numpy arrays or tensors with multiple values
                if hasattr(value, "item") and value.size == 1:
                    # For single-item arrays/tensors
                    log_dict[key] = value.item()
                else:
                    # For multi-dimensional arrays, convert to numpy and try to log
                    try:
                        if hasattr(value, "numpy"):
                            # For frameworks like PyTorch/TensorFlow
                            log_dict[key] = value.numpy()
                        else:
                            log_dict[key] = value
                    except:
                        # If conversion fails, skip this key
                        logger.warning(
                            f"Skipping logging of {key}: unsupported type {type(value)}"
                        )
            else:
                # Try to log other values, but this might fail for complex types
                try:
                    log_dict[key] = value
                except:
                    logger.warning(
                        f"Skipping logging of {key}: unsupported type {type(value)}"
                    )

        # Add global step to the logged metrics
        log_dict["global_step"] = step

        # Log to W&B with the current global step
        wandb.log(log_dict, step=step)

        # Update global step
        self.global_step = step

    def get_global_step(self) -> int:
        """Return current global step."""
        return self.global_step

    def set_global_step(self, step: int) -> None:
        """Set the global step to a specific value."""
        self.global_step = step

    def close(self) -> None:
        """Close the W&B run."""
        if not self._is_closed:
            wandb.finish()
            self._is_closed = True


class ContinuousWandbLogger(Logger):
    """
    Custom logger that maintains continuous step counting across training sessions
    when logging to Weights & Biases.

    This logger is designed to be used with Stable Baselines 3 algorithms
    to prevent the global step from resetting between training iterations.

    :param folder: the logging folder
    :param project: W&B project name
    :param run_name: W&B run name
    :param global_step_offset: Starting global step value to continue from
    :param config: Additional config values to log to W&B
    :param run_id: Run ID to resume a previous W&B run
    :param output_formats: Additional output formats beyond W&B
    """

    def __init__(
        self,
        folder: Optional[str] = None,
        global_step_offset: int = 0,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        output_formats: Optional[list] = None,
    ):
        # Create WandbOutputFormat
        wandb_format = WandbOutputFormat(
            global_step_offset=global_step_offset,
            config=config,
            run_id=run_id,
        )

        # Combine with any additional output formats
        formats = [wandb_format]
        if output_formats:
            formats.extend(output_formats)

        # Initialize parent class
        super().__init__(folder=folder, output_formats=formats)

        # Store reference to wandb format for direct access
        self.wandb_format = wandb_format

    def get_global_step(self) -> int:
        """Return current global step from the W&B output format."""
        return self.wandb_format.get_global_step()

    def set_global_step(self, step: int) -> None:
        """Set the global step to a specific value."""
        self.wandb_format.set_global_step(step)


# Function to create a logger instance
def create_continuous_wandb_logger(
    folder: Optional[str] = None,
    global_step_offset: int = 0,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    additional_formats: Optional[list[str]] = None,
) -> ContinuousWandbLogger:
    """
    Create a ContinuousWandbLogger with the specified parameters.

    :param folder: the logging folder
    :param project: W&B project name
    :param run_name: W&B run name
    :param global_step_offset: Starting global step value to continue from
    :param config: Additional config values to log to W&B
    :param run_id: Run ID to resume a previous W&B run
    :param formats: Additional output format strings beyond W&B (e.g., ["stdout", "csv"])
    :return: A configured ContinuousWandbLogger
    """
    from stable_baselines3.common.logger import make_output_format

    # Set up additional output formats if specified
    additional_log_formats = []
    if additional_formats:
        for format_str in additional_formats:
            if (
                format_str.lower() != "wandb"
            ):  # Skip 'wandb' format as we handle it ourselves
                additional_log_formats.append(make_output_format(format_str, folder))

    # Create and return the logger
    return ContinuousWandbLogger(
        folder=folder,
        global_step_offset=global_step_offset,
        config=config,
        run_id=run_id,
        output_formats=additional_log_formats,
    )
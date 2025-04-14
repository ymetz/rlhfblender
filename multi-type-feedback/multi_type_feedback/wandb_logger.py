import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class ContinuousWandbLogger(WandbLogger):
    """
    Custom WandbLogger that doesn't advance steps on its own,
    but instead logs at specific step points provided externally.

    This is specifically designed to work with RL training
    where step counting is managed by the RL algorithm.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment=None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            id=id,
            anonymous=anonymous,
            version=version,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            **kwargs,
        )
        self._cumulative_feedback = {}

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics at a specific step.

        Unlike default WandbLogger, this version doesn't increment step itself
        but requires a step to be provided.
        """
        # Only log if step is provided, otherwise do nothing
        if step is not None:
            super().log_metrics(metrics, step)

    @rank_zero_only
    def log_reward_metrics(
        self, reward_metrics: Dict[str, float] = None, feedback_counts: Dict[str, int] = None, step: Optional[int] = None
    ) -> None:
        """
        Log reward model metrics and feedback statistics at a specific step.

        Args:
            reward_metrics: Dictionary of reward model metrics (loss values)
            feedback_counts: Dictionary of feedback type counts
            step: Global step to use for logging
        """
        metrics_to_log = {}

        if reward_metrics:
            for feedback_type, loss in reward_metrics.items():
                metrics_to_log[f"reward_model/{feedback_type}_loss"] = loss

        if feedback_counts:
            for feedback_type, count in feedback_counts.items():
                metrics_to_log[f"feedback/{feedback_type}_count"] = count

                # Track cumulative counts
                if feedback_type not in self._cumulative_feedback:
                    self._cumulative_feedback[feedback_type] = 0
                self._cumulative_feedback[feedback_type] += count
                metrics_to_log[f"feedback/{feedback_type}_cumulative"] = self._cumulative_feedback[feedback_type]

        if metrics_to_log and step is not None:
            # Log using provided step
            self.log_metrics(metrics_to_log, step)

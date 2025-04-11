import os
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Literal, Dict, Any
from wandb.sdk.lib import telemetry as wb_telemetry
import logging

logger = logging.getLogger(__name__)

class ContinuousWandbCallback(BaseCallback):
    """
    Enhanced WandbCallback that preserves global step count across training iterations.
    
    This callback extends the functionality of the standard SB3 WandbCallback by:
    1. Maintaining a global step counter across multiple training iterations
    2. Supporting iterative training without step resets
    3. Adding custom metrics logging from reward models and feedback
    
    Args:
        verbose: Verbosity level for SB3 output
        model_save_path: Path to save models, None disables model saving
        model_save_freq: How often to save the model (in steps)
        gradient_save_freq: How often to log gradients (in steps)
        log: What to log. One of "gradients", "parameters", or "all"
        global_step_offset: Starting global step value to continue from
    """
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        global_step_offset: int = 0,
    ) -> None:
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before ContinuousWandbCallback()")
            
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
            
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.gradient_save_freq = gradient_save_freq
        self.global_step = global_step_offset  # Start from provided offset
        
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn(
                "`log` must be one of `None`, 'gradients', 'parameters', or 'all', "
                "falling back to 'all'"
            )
            log = "all"
        self.log = log
        
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you must set the `model_save_path` parameter"
        
        # Custom storage for reward model metrics and feedback stats
        self.reward_metrics = {}
        self.feedback_counts = {}
        self.cumulative_feedback = {}
        
    def _init_callback(self) -> None:
        """Initialize callback and log hyperparameters."""
        d = {}
        d["algo"] = type(self.model).__name__
        
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
                
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
            )
            
        wandb.config.setdefaults(d)
        
    def _on_step(self) -> bool:
        """Log training metrics and save model if needed."""
        # Get metrics directly from the SB3 logger
        print("WHAT IS THE LOGGER", self.model.logger, self.model.logger.output_formats)
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            metrics = self.model.logger.name_to_value
            print("METRICS", metrics)
            if metrics:
                # Only log scalar values to wandb
                wandb_metrics = {
                    key: value for key, value in metrics.items() 
                    if isinstance(value, (int, float))
                }
                
                # Add the global step to the metrics
                wandb_metrics["trainer/global_step"] = self.global_step
                
                if wandb_metrics:
                    wandb.log(wandb_metrics, step=self.global_step)
        
        # Save model if needed
        if self.model_save_freq > 0 and self.model_save_path is not None:
            if self.global_step % self.model_save_freq == 0:
                self.save_model()
                
        # Increment global step
        self.global_step += self.model.n_envs
        
        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.model_save_path is not None:
            self.save_model()
    
    def get_current_global_step(self) -> int:
        """Return current global step for external use."""
        return self.global_step
        
    def save_model(self) -> None:
        """Save the model and upload to W&B."""
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {self.path}")
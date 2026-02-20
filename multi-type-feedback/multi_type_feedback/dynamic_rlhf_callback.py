import os
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback


class RewardModelUpdateCallback(EventCallback):
    """
    Callback for updating reward models every N timesteps during RL training.

    This callback collects trajectories, gets feedback, updates reward models,
    and then lets training continue with the updated reward functions.

    :param drlhf_agent: The DynamicRLHF agent instance
    :param update_freq: Number of timesteps between updates
    :param sampling_strategy: Strategy for sampling feedback ("random" or "uncertainty")
    :param query_sampling_strategy: Strategy for query selection ("none", "average", "min", "max")
    :param query_sampling_multiplier: Multiplier for number of queries to sample before filtering
    :param verbose: Verbosity level: 0 for no output, 1 for info messages
    """

    def __init__(
        self,
        drlhf_agent,
        update_freq: int = 5000,
        sampling_strategy: str = "random",
        query_sampling_strategy: str = "none",
        query_sampling_multiplier: float = 2.0,
        verbose: int = 0,
    ):
        super().__init__(callback=None, verbose=verbose)
        self.drlhf_agent = drlhf_agent
        self.update_freq = update_freq
        self.sampling_strategy = sampling_strategy
        self.query_sampling_strategy = query_sampling_strategy
        self.query_sampling_multiplier = query_sampling_multiplier
        self.last_update_timestep = 0
        self.update_count = 0

    def _on_step(self) -> bool:
        """
        Check if it's time to update reward models and do so if needed.

        :return: True to continue training, False to stop
        """
        # Check if we need to update the reward models        
        timesteps_since_update = self.num_timesteps - self.last_update_timestep

        if timesteps_since_update >= self.update_freq:
            self.update_count += 1

            if self.verbose > 0:
                print(
                    f"\nReward model update #{self.update_count} at timestep {self.num_timesteps}"
                )

            # Determine number of trajectories to collect
            n_feedback_needed = self.drlhf_agent.n_feedback_per_iteration
            
            if self.query_sampling_strategy != "none":
                # Collect more trajectories for query selection
                n_trajectories_to_collect = int(n_feedback_needed * self.query_sampling_multiplier)
                if self.verbose > 0:
                    print(f"Collecting {n_trajectories_to_collect} trajectories for query selection (need {n_feedback_needed})")
            else:
                n_trajectories_to_collect = n_feedback_needed

            # Collect trajectories
            trajectories, initial_states = self.drlhf_agent.collect_trajectories(
                n_trajectories_to_collect
            )

            # Apply query selection if enabled
            if self.query_sampling_strategy != "none" and len(trajectories) > n_feedback_needed:
                if self.verbose > 0:
                    print(f"Selecting top {n_feedback_needed} queries using {self.query_sampling_strategy} strategy")
                
                # Select queries based on uncertainty
                selected_trajectories, selected_initial_states = (
                    self.drlhf_agent.select_queries_by_uncertainty(
                        trajectories, initial_states, n_feedback_needed, self.query_sampling_strategy
                    )
                )
                
                # Log basic query selection metrics
                if (
                    hasattr(self.drlhf_agent, "wandb")
                    and self.drlhf_agent.wandb.run is not None
                ):
                    query_metrics = {
                        "query_selection/total_collected": len(trajectories),
                        "query_selection/selected": len(selected_trajectories),
                    }
                    self.drlhf_agent.wandb.log(query_metrics, step=self.num_timesteps)
                
                trajectories, initial_states = selected_trajectories, selected_initial_states

            # Get feedback based on sampling strategy
            if self.sampling_strategy == "random":
                feedback, feedback_counts = self.drlhf_agent.sample_feedback_random(
                    trajectories, initial_states
                )
            else:  # uncertainty
                feedback, feedback_counts = (
                    self.drlhf_agent.sample_feedback_uncertainty(
                        trajectories, initial_states
                    )
                )

            # Update feedback buffers
            self.drlhf_agent.update_feedback_buffers(feedback)

            # Train reward models
            reward_metrics = self.drlhf_agent.train_reward_models()

            # Log metrics
            global_step = self.num_timesteps

            # Log via wandb if available
            if self.drlhf_agent.wandb_logger is not None:
                if hasattr(self.drlhf_agent.wandb_logger, "log_reward_metrics"):
                    self.drlhf_agent.wandb_logger.log_reward_metrics(
                        reward_metrics=reward_metrics,
                        feedback_counts=feedback_counts,
                        step=global_step,
                    )
                elif (
                    hasattr(self.drlhf_agent, "wandb")
                    and self.drlhf_agent.wandb.run is not None
                ):
                    # Fallback logging directly to wandb
                    metrics_to_log = {}
                    for feedback_type, loss in reward_metrics.items():
                        metrics_to_log[f"reward_model/{feedback_type}_loss"] = loss
                    for feedback_type, count in feedback_counts.items():
                        metrics_to_log[f"feedback/{feedback_type}_count"] = count
                    self.drlhf_agent.wandb.log(metrics_to_log, step=global_step)

            if self.verbose > 0:
                print("\nFeedback counts:")
                for feedback_type, count in feedback_counts.items():
                    print(f"{feedback_type}: {count}")

                print("\nReward model losses:")
                for feedback_type, loss in reward_metrics.items():
                    print(f"{feedback_type}: {loss:.4f}")

            # Compute and log extended evaluation metrics after each iteration
            try:
                if hasattr(self.drlhf_agent, "_log_iteration_metrics"):
                    self.drlhf_agent._log_iteration_metrics(step=global_step)
            except Exception as e:
                print(f"[WARN] Extended metric logging failed: {e}")

            # Update the last update timestep
            self.last_update_timestep = self.num_timesteps

        return True
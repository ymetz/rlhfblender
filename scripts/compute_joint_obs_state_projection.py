#!/usr/bin/env python3
"""
Compute Joint Observation and State Projections

This script combines functionality from compute_joint_projection.py and 
train_joint_observation_state_projection.py to efficiently compute both:
1. Joint observation projections across multiple checkpoints  
2. Inverse state projections that map coordinates back to environment states

The script saves intermediate results so you can use them independently:
- Joint observation projection results
- Trained inverse state projection model
- Combined metadata for both projections
"""

import argparse
import asyncio
import json
import logging
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rlhfblender.projections.inverse_state_projection_handler import (
    InverseStateProjectionHandler,
    collect_states_from_multiple_checkpoints
)
from compute_joint_projection import JointProjectionComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for saving results
JOINT_PROJECTIONS_DIR = Path("data/saved_projections/joint")
JOINT_PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

JOINT_OBS_STATE_DIR = Path("data/saved_projections/joint_obs_state")
JOINT_OBS_STATE_DIR.mkdir(parents=True, exist_ok=True)


class JointObservationStateProjectionComputer:
    """Computes both observation and state projections jointly."""
    
    def __init__(self,
                 experiment_name: str,
                 checkpoints: List[int],
                 projection_method: str = "PCA",
                 projection_props: Dict[str, Any] = None,
                 sequence_length: int = 1,
                 step_range: str = "[]",
                 reproject: bool = False,
                 append_time: bool = False,
                 transition_embedding: bool = True,
                 feature_embedding: bool = True,
                 additional_gym_packages: List[str] = None,
                 max_trajectories_per_checkpoint: int = 50,
                 max_steps_per_trajectory: int = 200,
                 state_network_params: Dict[str, Any] = None):
        
        self.experiment_name = experiment_name
        self.checkpoints = checkpoints
        self.projection_method = projection_method
        self.projection_props = projection_props or {}
        self.sequence_length = sequence_length
        self.step_range = step_range
        self.reproject = reproject
        self.append_time = append_time
        self.transition_embedding = transition_embedding
        self.feature_embedding = feature_embedding
        self.additional_gym_packages = additional_gym_packages or []
        self.max_trajectories_per_checkpoint = max_trajectories_per_checkpoint
        self.max_steps_per_trajectory = max_steps_per_trajectory
        
        self.state_network_params = state_network_params or {
            'hidden_dims': [128, 256, 256, 128],
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 64
        }
        
        # Will be filled during execution
        self.observation_computer = None
        self.state_projector = None
        self.env_name = None
        self.environment_config = None
    
    async def compute_observation_projection(self) -> Dict[str, Any]:
        """Compute joint observation projection across checkpoints."""
        logger.info("Computing joint observation projection...")
        
        # Create and configure observation projection computer
        self.observation_computer = JointProjectionComputer(
            experiment_name=self.experiment_name,
            checkpoints=self.checkpoints,
            projection_method=self.projection_method,
            projection_props=self.projection_props,
            sequence_length=self.sequence_length,
            step_range=self.step_range,
            reproject=self.reproject,
            append_time=self.append_time,
            transition_embedding=self.transition_embedding,
            feature_embedding=self.feature_embedding,
            additional_gym_packages=self.additional_gym_packages
        )
        
        # Load experiment info
        await self.observation_computer.load_experiment_info()
        self.env_name = self.observation_computer.env_name
        self.environment_config = self.observation_computer.environment_config
        
        # Load all episode data
        episode_data_list, checkpoint_episode_counts = await self.observation_computer.load_all_episode_data()
        
        # Check if we have any data
        if all(count == 0 for count in checkpoint_episode_counts):
            raise ValueError("No episode data found for any checkpoint!")
        
        # Preprocess and combine data
        combined_data, data_split_indices = self.observation_computer.preprocess_all_data(episode_data_list)
        
        # Fit joint projection
        handler = self.observation_computer.fit_joint_projection(combined_data)
        
        # Get projected data
        projected_data = handler.get_state()
        
        # Split projected data back to per-checkpoint
        checkpoint_projections = self.observation_computer.split_projected_data(projected_data, data_split_indices)
        
        # Save observation projection results
        obs_metadata_path = self.observation_computer.save_joint_projection(handler, checkpoint_projections)
        
        logger.info(f"Observation projection completed and saved to: {obs_metadata_path}")
        
        return {
            'handler': handler,
            'projected_data': projected_data,
            'checkpoint_projections': checkpoint_projections,
            'metadata_path': obs_metadata_path,
            'combined_data_shape': combined_data.shape,
            'data_split_indices': data_split_indices
        }
    
    def compute_state_projection(self, observation_coords: np.ndarray) -> Dict[str, Any]:
        """Compute inverse state projection using coordinates from observation projection."""
        logger.info("Computing inverse state projection...")
        
        # Collect environment states from the same checkpoints
        logger.info("Loading environment states...")
        all_states, _, checkpoint_indices = collect_states_from_multiple_checkpoints(
            db_experiment=self.experiment_name,
            checkpoints=self.checkpoints,
            environment_name=self.env_name,
            environment_config=self.environment_config,
            max_trajectories_per_checkpoint=self.max_trajectories_per_checkpoint,
            max_steps_per_trajectory=self.max_steps_per_trajectory,
            additional_gym_packages=self.additional_gym_packages
        )
        
        if not all_states:
            raise ValueError("No environment states could be loaded!")
        
        # Match states to observation coordinates
        logger.info("Matching states to coordinates...")
        min_length = min(len(all_states), len(observation_coords))
        matched_states = all_states[:min_length]
        matched_coords = observation_coords[:min_length]
        
        logger.info(f"Matched {min_length} state-coordinate pairs")
        
        # Initialize and train inverse state projection
        logger.info("Training inverse state projection...")
        self.state_projector = InverseStateProjectionHandler(
            hidden_dims=self.state_network_params['hidden_dims'],
            learning_rate=self.state_network_params['learning_rate'],
            num_epochs=self.state_network_params['num_epochs'],
            batch_size=self.state_network_params['batch_size']
        )
        
        state_training_history = self.state_projector.fit(
            coords=matched_coords,
            states=matched_states,
            validation_split=0.2
        )
        
        logger.info("Inverse state projection training completed")
        
        # Evaluate consistency
        consistency_metrics = self._evaluate_consistency(matched_coords, matched_states)
        
        return {
            'training_history': state_training_history,
            'consistency_metrics': consistency_metrics,
            'matched_data_size': min_length,
            'total_states_found': len(all_states),
            'state_projector': self.state_projector
        }
    
    def _evaluate_consistency(self, coords: np.ndarray, true_states: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate consistency between coordinates and reconstructed states."""
        logger.info("Evaluating coordinate-state consistency...")
        
        # Sample a subset for evaluation
        n_eval = min(1000, len(coords))
        eval_indices = np.random.choice(len(coords), n_eval, replace=False)
        eval_coords = coords[eval_indices]
        eval_states = [true_states[i] for i in eval_indices]
        
        # Predict states from coordinates
        predicted_states = self.state_projector.predict(eval_coords)
        
        # Compute reconstruction error for each state component
        component_errors = {}
        total_mse = 0.0
        n_components = 0
        
        for true_state, pred_state in zip(eval_states, predicted_states):
            # Handle the case where true_state might be an array from VecEnv (size 1)
            if isinstance(true_state, np.ndarray) and len(true_state) == 1:
                true_state = true_state[0]
            
            # Skip if true_state is None or not a dict
            if true_state is None or not isinstance(true_state, dict):
                continue
                
            for key in true_state.keys():
                if key in pred_state:
                    true_val = np.array(true_state[key]).flatten()
                    pred_val = np.array(pred_state[key]).flatten()
                    
                    if len(true_val) == len(pred_val):
                        mse = np.mean((true_val - pred_val) ** 2)
                        
                        if key not in component_errors:
                            component_errors[key] = []
                        component_errors[key].append(mse)
                        
                        total_mse += mse
                        n_components += 1
        
        # Average errors
        avg_component_errors = {
            key: float(np.mean(errors)) 
            for key, errors in component_errors.items()
        }
        overall_mse = total_mse / n_components if n_components > 0 else float('inf')
        
        metrics = {
            'overall_mse': float(overall_mse),
            'component_errors': avg_component_errors,
            'evaluation_samples': n_eval
        }
        
        logger.info(f"Consistency evaluation - Overall MSE: {overall_mse:.6f}")
        return metrics
    
    def save_joint_results(self, obs_results: Dict[str, Any], state_results: Dict[str, Any]) -> str:
        """Save combined observation and state projection results."""
        # Create filename for combined results
        joint_filename = f"{self.env_name}_{self.observation_computer.db_experiment.id}_joint_obs_state_{self.projection_method}_{min(self.checkpoints)}_{max(self.checkpoints)}"
        
        # Save the inverse state projection model
        state_model_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_state_model.pkl"
        self.state_projector.save_model(str(state_model_path))
        logger.info(f"Saved state projection model to: {state_model_path}")
        
        # Save combined results
        results_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_results.npz"
        save_dict = {
            'checkpoints': np.array(self.checkpoints),
            'observation_coords': obs_results['projected_data'],
            'data_split_indices': np.array(obs_results['data_split_indices']),
        }
        
        # Add per-checkpoint observation projections
        for cp, proj in obs_results['checkpoint_projections'].items():
            save_dict[f"obs_checkpoint_{cp}"] = proj
        
        np.savez(results_path, **save_dict)
        logger.info(f"Saved combined results to: {results_path}")
        
        # Save comprehensive metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.observation_computer.db_experiment.id,
            "env_name": self.env_name,
            "checkpoints": self.checkpoints,
            "projection_method": self.projection_method,
            "projection_props": self.projection_props,
            "sequence_length": self.sequence_length,
            "step_range": self.step_range,
            "reproject": self.reproject,
            "append_time": self.append_time,
            "transition_embedding": self.transition_embedding,
            "feature_embedding": self.feature_embedding,
            "max_trajectories_per_checkpoint": self.max_trajectories_per_checkpoint,
            "max_steps_per_trajectory": self.max_steps_per_trajectory,
            "state_network_params": self.state_network_params,
            # Paths
            "observation_projection_metadata": obs_results['metadata_path'],
            "state_model_path": str(state_model_path),
            "combined_results_path": str(results_path),
            # Results summary
            "observation_data_shape": obs_results['combined_data_shape'],
            "state_consistency_mse": state_results['consistency_metrics']['overall_mse'],
            "matched_state_coordinate_pairs": state_results['matched_data_size'],
            "total_states_found": state_results['total_states_found']
        }
        
        metadata_path = JOINT_OBS_STATE_DIR / f"{joint_filename}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved combined metadata to: {metadata_path}")
        
        return str(metadata_path)
    
    async def compute_joint_projections(self) -> str:
        """
        Main method to compute both observation and state projections.
        
        Returns:
            Path to combined metadata file
        """
        try:
            logger.info(f"Starting joint observation-state projection computation for {len(self.checkpoints)} checkpoints")
            logger.info(f"Checkpoints: {self.checkpoints}")
            logger.info(f"Projection method: {self.projection_method}")
            
            # Step 1: Compute observation projection
            obs_results = await self.compute_observation_projection()
            
            # Step 2: Compute state projection using observation coordinates
            state_results = self.compute_state_projection(obs_results['projected_data'])
            
            # Step 3: Save combined results
            metadata_path = self.save_joint_results(obs_results, state_results)
            
            logger.info("Joint observation-state projection computation completed successfully!")
            logger.info(f"Combined metadata saved to: {metadata_path}")
            
            # Print summary
            logger.info("\nSummary:")
            logger.info(f"  Observation data shape: {obs_results['combined_data_shape']}")
            logger.info(f"  State consistency MSE: {state_results['consistency_metrics']['overall_mse']:.6f}")
            logger.info(f"  Matched pairs: {state_results['matched_data_size']}")
            
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error computing joint projections: {e}")
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description="Compute joint observation and state projections")
    
    # Required arguments
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True,
                       help="List of checkpoint steps (e.g., --checkpoints 1000 2000 3000)")
    
    # Projection parameters
    parser.add_argument("--projection-method", type=str, default="PCA", 
                       help="Projection method (UMAP, PCA, t-SNE, etc.)")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length for projection")
    parser.add_argument("--step-range", type=str, default="[]", help="Range of steps to include, e.g. '[0,100]'")
    parser.add_argument("--reproject", action="store_true", help="Whether to reproject observations")
    parser.add_argument("--append-time", action="store_true", help="Append time information")
    parser.add_argument("--no-transition", action="store_true", help="Skip transition embedding computation")
    parser.add_argument("--no-feature", action="store_true", help="Skip feature embedding computation")
    
    # Data loading parameters
    parser.add_argument("--max-trajectories", type=int, default=50, 
                       help="Max trajectories per checkpoint for state collection")
    parser.add_argument("--max-steps", type=int, default=200, 
                       help="Max steps per trajectory for state collection")
    parser.add_argument("--additional-gym-packages", type=str, nargs="+", default=[], 
                       help="Additional gym packages to import")
    
    # State projection parameters
    parser.add_argument("--state-epochs", type=int, default=100, help="Training epochs for state projection")
    parser.add_argument("--state-batch-size", type=int, default=64, help="Batch size for state projection")
    parser.add_argument("--state-learning-rate", type=float, default=0.001, help="Learning rate for state projection")
    
    # Projection method specific parameters
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric")
    
    args = parser.parse_args()
    
    # Checkpoints are already parsed as integers by argparse
    checkpoints = args.checkpoints
    
    if len(checkpoints) < 1:
        logger.error("At least 1 checkpoint is required.")
        sys.exit(1)
    
    logger.info(f"Computing joint projections for {len(checkpoints)} checkpoints: {checkpoints}")
    
    # Build projection properties
    projection_props = {"n_components": 2}  # Always 2D for visualization
    
    if args.projection_method == "UMAP":
        projection_props.update({
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
        })
    elif args.projection_method == "t-SNE":
        projection_props.update({
            "perplexity": 30,
            "early_exaggeration": 12,
            "learning_rate": 200,
        })
    
    # Create joint projection computer
    computer = JointObservationStateProjectionComputer(
        experiment_name=args.experiment_name,
        checkpoints=checkpoints,
        projection_method=args.projection_method,
        projection_props=projection_props,
        sequence_length=args.sequence_length,
        step_range=args.step_range,
        reproject=args.reproject,
        append_time=args.append_time,
        transition_embedding=not args.no_transition,
        feature_embedding=not args.no_feature,
        additional_gym_packages=args.additional_gym_packages,
        max_trajectories_per_checkpoint=args.max_trajectories,
        max_steps_per_trajectory=args.max_steps,
        state_network_params={
            'hidden_dims': [128, 256, 256, 128],
            'learning_rate': args.state_learning_rate,
            'num_epochs': args.state_epochs,
            'batch_size': args.state_batch_size
        }
    )
    
    # Run computation
    try:
        metadata_path = asyncio.run(computer.compute_joint_projections())
        print(f"\nSuccess! Joint projections completed.")
        print(f"Combined metadata file: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Failed to compute joint projections: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
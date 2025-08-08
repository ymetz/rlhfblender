#!/usr/bin/env python3
"""
Train Joint Observation-State Projection

This script trains both a forward observation projection and an inverse state projection
simultaneously to ensure consistency between coordinate space and state space.

The approach:
1. Load episode data from multiple checkpoints
2. Collect both observations and environment states
3. Train a joint projection that maps observations -> 2D coordinates
4. Train an inverse projection that maps 2D coordinates -> states
5. Use joint loss to ensure consistency between forward and inverse mappings
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pickle
import torch

from rlhfblender.projections.inverse_state_projection_handler import (
    InverseStateProjectionHandler, 
    collect_states_from_multiple_checkpoints
)
from compute_joint_projection import JointProjectionComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JointObservationStateProjection:
    """Joint training of observation projection and state inverse projection."""
    
    def __init__(self,
                 projection_method: str = "UMAP",
                 state_network_params: Dict[str, Any] = None,
                 joint_loss_weight: float = 0.1,
                 device: str = "auto"):
        
        self.projection_method = projection_method
        self.state_network_params = state_network_params or {
            'hidden_dims': [128, 256, 256, 128],
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 64
        }
        self.joint_loss_weight = joint_loss_weight
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Components
        self.observation_projector = None
        self.state_inverse_projector = None
        
        logger.info(f"Using device: {self.device}")
    
    def fit_joint_projection(self,
                           experiment: str,
                           checkpoints: List[int],
                           additional_gym_packages: List[str] = [],
                           max_trajectories_per_checkpoint: int = 50,
                           max_steps_per_trajectory: int = 200) -> Dict[str, Any]:
        """
        Train joint observation projection and state inverse projection.
        
        Args:
            experiment: Database experiment Name
            checkpoints: List of checkpoint steps to include
            max_trajectories_per_checkpoint: Max trajectories per checkpoint
            max_steps_per_trajectory: Max steps per trajectory
            
        Returns:
            Training history and results
        """
        import asyncio
        
        async def _fit_joint_projection_async():
            logger.info("Starting joint observation-state projection training...")
            logger.info(f"Checkpoints: {checkpoints}")
            
            # Step 1: Train forward observation projection using existing joint projection approach
            logger.info("Step 1: Training forward observation projection...")
            joint_computer = JointProjectionComputer(
                experiment_name=experiment,
                checkpoints=checkpoints,
                projection_method=self.projection_method,
                sequence_length=1,
                step_range="[]",
                append_time=False,
                projection_props={},
                additional_gym_packages=additional_gym_packages,
            )
            
            # Load experiment info
            await joint_computer.load_experiment_info()
            logger.info(f"Environment: {joint_computer.env_name}")
            
            # Load episode data from all checkpoints
            episode_data_list, checkpoint_episode_counts = await joint_computer.load_all_episode_data()
            
            # Preprocess and combine data into numpy array
            combined_data, split_indices = joint_computer.preprocess_all_data(episode_data_list)
            
            # Fit joint projection on combined data
            projection_handler = joint_computer.fit_joint_projection(combined_data)
            
            # Transform data to get coordinates
            coordinates = projection_handler.transform(combined_data)
            
            return {
                'handler': projection_handler,
                'coordinates': coordinates,
                'split_indices': split_indices,
                'environment_name': joint_computer.env_name,
                "env_config": joint_computer.environment_config
            }
        
        # Run the async part
        forward_projection = asyncio.run(_fit_joint_projection_async())

        # load db_EXPERIMENT FROM database

        
        self.observation_projector = forward_projection['handler']
        observation_coords = forward_projection['coordinates']  # Combined coordinates from all checkpoints
        environment_name = forward_projection['environment_name']
        env_config = forward_projection['env_config']
        
        logger.info(f"Forward projection completed. Shape: {observation_coords.shape}")
        
        # Step 2: Collect environment states from the same data
        logger.info("Step 2: Collecting environment states...")
        all_states, dummy_coords, checkpoint_indices = collect_states_from_multiple_checkpoints(
            db_experiment=experiment,
            checkpoints=checkpoints,
            environment_name=environment_name,
            environment_config=env_config,
            max_trajectories_per_checkpoint=max_trajectories_per_checkpoint,
            max_steps_per_trajectory=max_steps_per_trajectory,
            additional_gym_packages=additional_gym_packages
        )
        
        # Step 3: Match states to observation coordinates
        logger.info("Step 3: Matching states to coordinates...")
        min_length = min(len(all_states), len(observation_coords))
        matched_states = all_states[:min_length]
        matched_coords = observation_coords[:min_length]
        matched_indices = checkpoint_indices[:min_length] if len(checkpoint_indices) >= min_length else checkpoint_indices
        
        logger.info(f"Matched {min_length} state-coordinate pairs")
        
        # Step 4: Train inverse state projection
        logger.info("Step 4: Training inverse state projection...")
        self.state_inverse_projector = InverseStateProjectionHandler(
            hidden_dims=self.state_network_params['hidden_dims'],
            learning_rate=self.state_network_params['learning_rate'],
            num_epochs=self.state_network_params['num_epochs'],
            batch_size=self.state_network_params['batch_size'],
            device=str(self.device)
        )
        
        state_training_history = self.state_inverse_projector.fit(
            coords=matched_coords,
            states=matched_states,
            validation_split=0.2
        )
        
        logger.info("Inverse state projection training completed")
        
        # Step 5: Evaluate consistency
        logger.info("Step 5: Evaluating forward-inverse consistency...")
        consistency_metrics = self._evaluate_consistency(
            matched_coords, matched_states
        )
        
        # Prepare results
        results = {
            'forward_projection': forward_projection,
            'inverse_state_history': state_training_history,
            'consistency_metrics': consistency_metrics,
            'matched_data_size': min_length,
            'checkpoints': checkpoints,
            'environment_name': environment_name
        }
        
        logger.info("Joint training completed successfully!")
        return results
    
    def _evaluate_consistency(self, 
                            coords: np.ndarray, 
                            true_states: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate consistency between forward and inverse projections.
        
        This measures how well the inverse projection can reconstruct states
        from coordinates that were generated by the forward projection.
        """
        logger.info("Evaluating forward-inverse consistency...")
        
        # Sample a subset for evaluation
        n_eval = min(1000, len(coords))
        eval_indices = np.random.choice(len(coords), n_eval, replace=False)
        eval_coords = coords[eval_indices]
        eval_states = [true_states[i] for i in eval_indices]
        
        # Predict states from coordinates
        predicted_states = self.state_inverse_projector.predict(eval_coords)
        
        # Compute reconstruction error for each state component
        component_errors = {}
        total_mse = 0.0
        n_components = 0
        
        for i, (true_state, pred_state) in enumerate(zip(eval_states, predicted_states)):
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
            key: np.mean(errors) 
            for key, errors in component_errors.items()
        }
        overall_mse = total_mse / n_components if n_components > 0 else float('inf')
        
        metrics = {
            'overall_mse': overall_mse,
            'component_errors': avg_component_errors,
            'evaluation_samples': n_eval
        }
        
        logger.info(f"Consistency evaluation - Overall MSE: {overall_mse:.6f}")
        return metrics
    
    def save_joint_model(self, output_dir: str):
        """Save both forward and inverse projection models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save forward projection (observation -> coordinates)
        forward_path = output_path / "forward_projection.pkl"
        with open(forward_path, 'wb') as f:
            pickle.dump(self.observation_projector, f)
        
        # Save inverse projection (coordinates -> states)
        inverse_path = output_path / "inverse_state_projection.pkl"
        self.state_inverse_projector.save_model(str(inverse_path))
        
        logger.info(f"Joint model saved to {output_dir}")
        return {
            'forward_path': str(forward_path),
            'inverse_path': str(inverse_path)
        }
    
    def load_joint_model(self, model_dir: str):
        """Load both forward and inverse projection models."""
        model_path = Path(model_dir)
        
        # Load forward projection
        forward_path = model_path / "forward_projection.pkl"
        with open(forward_path, 'rb') as f:
            self.observation_projector = pickle.load(f)
        
        # Load inverse projection
        inverse_path = model_path / "inverse_state_projection.pkl"
        self.state_inverse_projector = InverseStateProjectionHandler()
        self.state_inverse_projector.load_model(str(inverse_path))
        
        logger.info(f"Joint model loaded from {model_dir}")
    
    def predict_state_from_observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: observation -> coordinates -> state.
        
        Args:
            observation: Single observation array
            
        Returns:
            Predicted environment state
        """
        if self.observation_projector is None or self.state_inverse_projector is None:
            raise ValueError("Models not trained. Call fit_joint_projection() first.")
        
        # Forward projection: observation -> coordinates
        coords = self.observation_projector.transform(observation.reshape(1, -1))
        
        # Inverse projection: coordinates -> state
        predicted_states = self.state_inverse_projector.predict(coords)
        
        return predicted_states[0]


def main():
    parser = argparse.ArgumentParser(description="Train joint observation-state projection")
    parser.add_argument("--experiment", "-e", required=True,
                       help="Database experiment Name")
    parser.add_argument("--checkpoints", "-c", nargs="+", type=int, required=True,
                       help="List of checkpoint steps (e.g., --checkpoints 100000 200000 300000)")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Output directory for trained models")
    parser.add_argument("--projection-method", default="PCA",
                       choices=["UMAP", "PCA", "TSNE"],
                       help="Forward projection method")
    parser.add_argument("--max-trajectories", type=int, default=50,
                       help="Max trajectories per checkpoint")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Max steps per trajectory")
    parser.add_argument("--state-epochs", type=int, default=100,
                       help="Training epochs for inverse state projection")
    parser.add_argument("--state-batch-size", type=int, default=64,
                       help="Batch size for inverse state projection")
    parser.add_argument("--joint-loss-weight", type=float, default=0.1,
                       help="Weight for joint consistency loss")
    parser.add_argument(
        "--additional-gym-packages",
        type=str,
        nargs="+",
        help="(Optional) Additional gym packages to import.",
        default=[],
    )
    
    args = parser.parse_args()
    
    # Create joint trainer
    trainer = JointObservationStateProjection(
        projection_method=args.projection_method,
        state_network_params={
            'hidden_dims': [128, 256, 256, 128],
            'learning_rate': 0.001,
            'num_epochs': args.state_epochs,
            'batch_size': args.state_batch_size
        },
        joint_loss_weight=args.joint_loss_weight
    )
    
    # Train joint projection
    results = trainer.fit_joint_projection(
        experiment=args.experiment,
        checkpoints=args.checkpoints,
        max_trajectories_per_checkpoint=args.max_trajectories,
        max_steps_per_trajectory=args.max_steps,
        additional_gym_packages=args.additional_gym_packages
    )
    
    # Save models
    model_paths = trainer.save_joint_model(args.output_dir)
    
    # Save training results
    results_path = Path(args.output_dir) / "training_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("Joint training completed successfully!")
    print(f"Models saved to: {args.output_dir}")
    print(f"Forward projection: {model_paths['forward_path']}")
    print(f"Inverse state projection: {model_paths['inverse_path']}")
    print(f"Training results: {results_path}")
    
    # Print consistency metrics
    consistency = results['consistency_metrics']
    print(f"\nConsistency Evaluation:")
    print(f"  Overall MSE: {consistency['overall_mse']:.6f}")
    print(f"  Component errors: {consistency['component_errors']}")


if __name__ == "__main__":
    main()
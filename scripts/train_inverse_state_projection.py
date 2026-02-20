#!/usr/bin/env python3
"""
Train Inverse State Projection Model

This script collects environment states from existing trajectory data and trains
a neural network to map 2D projection coordinates back to environment states.
This enables demo generation from novel clicked coordinates.
"""

import argparse
import logging
import numpy as np
import os
from typing import List, Dict
import pickle
import glob

from rlhfblender.data_collection.environment_handler import get_environment
from rlhfblender.projections.inverse_state_projection_handler import (
    InverseStateProjectionHandler, 
    collect_trajectory_states
)
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_trajectory_files(data_dir: str, environment_name: str) -> List[str]:
    """Find all trajectory .npz files for a given environment."""
    pattern = os.path.join(data_dir, "episodes", environment_name, "**", "*.npz")
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(files)} trajectory files for {environment_name}")
    return files


def load_projection_data(projection_path: str) -> Dict[str, np.ndarray]:
    """Load existing projection coordinates."""
    with open(projection_path, 'rb') as f:
        projection_data = pickle.load(f)
    return projection_data


def collect_states_from_trajectories(trajectory_files: List[str], 
                                   env_wrapper,
                                   max_trajectories: int = None,
                                   max_steps_per_trajectory: int = None) -> List[Dict[str, np.ndarray]]:
    """
    Collect environment states by replaying trajectories.
    
    Args:
        trajectory_files: List of .npz file paths
        env_wrapper: SaveResetWrapper environment
        max_trajectories: Limit number of trajectories to process
        max_steps_per_trajectory: Limit steps per trajectory
        
    Returns:
        List of state dictionaries
    """
    all_states = []
    
    files_to_process = trajectory_files[:max_trajectories] if max_trajectories else trajectory_files
    
    for i, filepath in enumerate(files_to_process):
        logger.info(f"Processing trajectory {i+1}/{len(files_to_process)}: {filepath}")
        
        try:
            # Load trajectory data
            episode_data = np.load(filepath)
            
            # Limit steps if specified
            if max_steps_per_trajectory:
                for key in ['obs', 'actions']:
                    if key in episode_data:
                        episode_data[key] = episode_data[key][:max_steps_per_trajectory]
            
            # Create dummy coordinates (will be replaced with actual projection coords)
            n_steps = len(episode_data['obs'])
            dummy_coords = np.random.randn(n_steps, 2)
            
            # Collect states from this trajectory
            states = collect_trajectory_states(episode_data, env_wrapper, dummy_coords)
            all_states.extend(states)
            
            logger.info(f"  Collected {len(states)} states")
            
        except Exception as e:
            logger.warning(f"  Failed to process {filepath}: {e}")
            continue
    
    logger.info(f"Total states collected: {len(all_states)}")
    return all_states


def match_states_to_coordinates(states: List[Dict[str, np.ndarray]], 
                               projection_data: Dict[str, np.ndarray],
                               trajectory_files: List[str]) -> tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Match collected states to their corresponding 2D projection coordinates.
    This is a simplified version - in practice you'd need more sophisticated matching.
    """
    # For now, assume states and coordinates are in the same order
    # In practice, you'd need to match based on trajectory IDs and step numbers
    
    coords = projection_data.get('coordinates', projection_data.get('coords'))
    if coords is None:
        raise ValueError("No coordinates found in projection data")
    
    # Truncate to match shorter array
    min_len = min(len(states), len(coords))
    matched_states = states[:min_len]
    matched_coords = coords[:min_len]
    
    logger.info(f"Matched {min_len} states to coordinates")
    return matched_states, matched_coords


def main():
    parser = argparse.ArgumentParser(description="Train inverse state projection model")
    parser.add_argument("--environment", "-e", required=True, 
                       help="Environment name (e.g., 'metaworld_reach-v2')")
    parser.add_argument("--data-dir", "-d", default="data", 
                       help="Data directory path")
    parser.add_argument("--projection-file", "-p", required=True,
                       help="Path to projection coordinates file (.pkl)")
    parser.add_argument("--output-model", "-o", required=True,
                       help="Output path for trained model")
    parser.add_argument("--max-trajectories", "-t", type=int, default=100,
                       help="Maximum trajectories to process")
    parser.add_argument("--max-steps", "-s", type=int, default=200,
                       help="Maximum steps per trajectory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.projection_file):
        raise FileNotFoundError(f"Projection file not found: {args.projection_file}")
    
    # Find trajectory files
    trajectory_files = find_trajectory_files(args.data_dir, args.environment)
    if not trajectory_files:
        raise ValueError(f"No trajectory files found for environment {args.environment}")
    
    # Load projection coordinates
    logger.info(f"Loading projection data from {args.projection_file}")
    projection_data = load_projection_data(args.projection_file)
    
    # Create environment with SaveResetWrapper
    logger.info(f"Creating environment: {args.environment}")
    base_env = get_environment(
        args.environment,
        environment_config={"render_mode": "rgb_array"},
        n_envs=1
    )
    
    # Wrap with SaveResetWrapper
    env = base_env.envs[0] if hasattr(base_env, 'envs') else base_env
    env_wrapper = SaveResetEnvWrapper(env)
    
    # Collect states from trajectories
    logger.info("Collecting states from trajectories...")
    states = collect_states_from_trajectories(
        trajectory_files, 
        env_wrapper,
        max_trajectories=args.max_trajectories,
        max_steps_per_trajectory=args.max_steps
    )
    
    # Match states to coordinates
    logger.info("Matching states to projection coordinates...")
    matched_states, matched_coords = match_states_to_coordinates(
        states, projection_data, trajectory_files
    )
    
    # Train inverse projection model
    logger.info("Training inverse state projection model...")
    handler = InverseStateProjectionHandler(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    history = handler.fit(matched_coords, matched_states)
    
    # Save model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    handler.save_model(args.output_model)
    
    # Save training history
    history_path = args.output_model.replace('.pkl', '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    logger.info(f"Model saved to: {args.output_model}")
    logger.info(f"Training history saved to: {history_path}")
    
    # Cleanup
    env_wrapper.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
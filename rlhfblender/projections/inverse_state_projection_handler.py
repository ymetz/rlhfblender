"""
Inverse State Projection Handler

Maps 2D coordinates back to actual environment states (not just observations).
This enables loading novel states for demo generation from clicked coordinates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from rlhfblender.data_collection.environment_handler import get_environment
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper
import os
import glob
import pickle
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class InverseStateProjectionNetwork(nn.Module):
    """Neural network that maps 2D coordinates to environment states."""
    
    def __init__(self, input_dim: int = 2, output_dim: int = None, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]
            
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if i > 0:  # Skip dropout on first layer
                x = self.dropout(x)
        
        # Output layer (no activation for regression)
        x = self.layers[-1](x)
        return x


class InverseStateProjectionHandler:
    """Handles training and inference for coordinate-to-state mapping."""
    
    def __init__(self, 
                 hidden_dims: List[int] = None,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 batch_size: int = 64,
                 device: str = "auto"):
        
        self.hidden_dims = hidden_dims or [128, 256, 256, 128]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.state_scaler = StandardScaler()
        self.coord_scaler = StandardScaler()
        self.state_structure = None
        self.is_fitted = False
        
        logger.info(f"Using device: {self.device}")
        
    def _flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten structured state dictionary to vector."""
        if self.state_structure is None:
            # Learn structure from first state
            self.state_structure = {}
            flat_state = []
            start_idx = 0
            
            for key, value in state.items():
                flat_val = np.array(value).flatten()
                self.state_structure[key] = {
                    'shape': np.array(value).shape,
                    'start_idx': start_idx,
                    'end_idx': start_idx + len(flat_val)
                }
                flat_state.extend(flat_val)
                start_idx += len(flat_val)
                
            return np.array(flat_state)
        else:
            # Use existing structure
            flat_state = []
            for key in self.state_structure.keys():
                if key in state:
                    flat_val = np.array(state[key]).flatten()
                    flat_state.extend(flat_val)
                else:
                    # Fill with zeros if key missing
                    shape = self.state_structure[key]['shape']
                    flat_val = np.zeros(np.prod(shape))
                    flat_state.extend(flat_val)
            return np.array(flat_state)
    
    def _unflatten_state(self, flat_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Reconstruct structured state dictionary from flat vector."""
        if self.state_structure is None:
            raise ValueError("State structure not learned. Call fit() first.")
            
        state = {}
        for key, info in self.state_structure.items():
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            shape = info['shape']
            
            flat_val = flat_state[start_idx:end_idx]
            state[key] = flat_val.reshape(shape)
            
        return state
    
    def fit(self, 
            coords: np.ndarray, 
            states: List[Dict[str, np.ndarray]], 
            validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the inverse state projection model.
        
        Args:
            coords: 2D coordinates, shape (N, 2)
            states: List of state dictionaries from SaveResetWrapper
            validation_split: Fraction of data for validation
            
        Returns:
            Training history with loss curves
        """
        logger.info(f"Training inverse state projection on {len(states)} states")
        
        # Convert states to flat vectors
        flat_states = np.array([self._flatten_state(state) for state in states])
        
        logger.info(f"State dimension: {flat_states.shape[1]}")
        logger.info(f"Coordinate dimension: {coords.shape[1]}")
        
        # Scale inputs and outputs
        coords_scaled = self.coord_scaler.fit_transform(coords)
        states_scaled = self.state_scaler.fit_transform(flat_states)
        
        # Create train/validation split
        n_samples = len(coords_scaled)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        coords_train = coords_scaled[train_idx]
        states_train = states_scaled[train_idx]
        coords_val = coords_scaled[val_idx]
        states_val = states_scaled[val_idx]
        
        # Initialize model
        self.model = InverseStateProjectionNetwork(
            input_dim=coords.shape[1],
            output_dim=flat_states.shape[1],
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        coords_train_tensor = torch.FloatTensor(coords_train).to(self.device)
        states_train_tensor = torch.FloatTensor(states_train).to(self.device)
        coords_val_tensor = torch.FloatTensor(coords_val).to(self.device)
        states_val_tensor = torch.FloatTensor(states_val).to(self.device)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            # Training phase
            epoch_train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(coords_train_tensor), self.batch_size):
                batch_coords = coords_train_tensor[i:i+self.batch_size]
                batch_states = states_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_coords)
                loss = criterion(predictions, batch_states)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_train_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(coords_val_tensor)
                val_loss = criterion(val_predictions, states_val_tensor).item()
                history['val_loss'].append(val_loss)
            
            self.model.train()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_fitted = True
        logger.info("Training completed successfully")
        
        return history
    
    def predict(self, coords: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Map 2D coordinates to environment states.
        
        Args:
            coords: 2D coordinates, shape (N, 2)
            
        Returns:
            List of state dictionaries compatible with SaveResetWrapper
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Scale coordinates
        coords_scaled = self.coord_scaler.transform(coords)
        coords_tensor = torch.FloatTensor(coords_scaled).to(self.device)
        
        # Predict states
        self.model.eval()
        with torch.no_grad():
            flat_states_scaled = self.model(coords_tensor).cpu().numpy()
        
        # Inverse transform
        flat_states = self.state_scaler.inverse_transform(flat_states_scaled)
        
        # Convert back to structured states
        states = []
        for flat_state in flat_states:
            state = self._unflatten_state(flat_state)
            states.append(state)
            
        return states
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        model_data = {
            'state_dict': self.model.state_dict(),
            'state_structure': self.state_structure,
            'state_scaler': self.state_scaler,
            'coord_scaler': self.coord_scaler,
            'hidden_dims': self.hidden_dims,
            'device': str(self.device)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scalers."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore structure and scalers
        self.state_structure = model_data['state_structure']
        self.state_scaler = model_data['state_scaler']
        self.coord_scaler = model_data['coord_scaler']
        self.hidden_dims = model_data['hidden_dims']
        
        # Reconstruct model
        state_dim = len(self.state_scaler.scale_) if hasattr(self.state_scaler, 'scale_') else None
        if state_dim is None:
            raise ValueError("Cannot determine state dimension from loaded scaler")
            
        self.model = InverseStateProjectionNetwork(
            input_dim=2,  # Always 2D coordinates
            output_dim=state_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.model.load_state_dict(model_data['state_dict'])
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


def collect_trajectory_states_from_episode(episode_data: Dict[str, np.ndarray], 
                                          env_wrapper) -> List[Dict[str, np.ndarray]]:
    """
    Collect environment states from a trajectory using SaveResetWrapper.
    
    Args:
        episode_data: Episode data from .npz file
        env_wrapper: Environment wrapped with SaveResetWrapper
        
    Returns:
        List of state dictionaries for each step
    """
    states = []
    observations = episode_data['obs']
    actions = episode_data['actions']
    
    # Reset environment
    env_wrapper.reset()
    
    # Replay trajectory and collect states
    for i in range(len(observations)):
        # Save current state
        state_data = env_wrapper.save_state(observation=observations[i])
        states.append(state_data['state'])  # Just the state part, not observation
        
        # Take action (if not at final step)
        if i < len(actions):
            env_wrapper.step(actions[i])
    
    return states


def collect_states_from_multiple_checkpoints(
    experiment_id: str,
    checkpoints: List[int],
    environment_name: str,
    max_trajectories_per_checkpoint: int = 50,
    max_steps_per_trajectory: int = 200
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, List[int]]:
    """
    Collect environment states from multiple checkpoints for joint training.
    
    Args:
        experiment_id: Database experiment ID
        checkpoints: List of checkpoint steps
        environment_name: Environment name
        max_trajectories_per_checkpoint: Max trajectories per checkpoint
        max_steps_per_trajectory: Max steps per trajectory
        
    Returns:
        Tuple of (all_states, all_coordinates, checkpoint_indices)
    """
    logger.info(f"Collecting states from {len(checkpoints)} checkpoints: {checkpoints}")
    
    all_states = []
    all_coordinates = []
    checkpoint_indices = []  # Track which checkpoint each state belongs to
    
    # Create environment
    base_env = get_environment(environment_name, environment_config={"render_mode": "rgb_array"}, n_envs=1)
    env = base_env.envs[0] if hasattr(base_env, 'envs') else base_env
    env_wrapper = SaveResetEnvWrapper(env)
    
    try:
        for checkpoint_idx, checkpoint in enumerate(checkpoints):
            logger.info(f"Processing checkpoint {checkpoint} ({checkpoint_idx+1}/{len(checkpoints)})")
            
            # Find trajectory files for this checkpoint
            data_pattern = os.path.join("data", "episodes", environment_name, 
                                      f"*{checkpoint}", "*.npz")
            trajectory_files = glob.glob(data_pattern)
            
            if not trajectory_files:
                logger.warning(f"No trajectory files found for checkpoint {checkpoint}")
                continue
                
            # Limit number of trajectories
            files_to_process = trajectory_files[:max_trajectories_per_checkpoint]
            
            for file_path in files_to_process:
                try:
                    # Load episode data
                    episode_data = np.load(file_path)
                    
                    # Limit steps if specified
                    if max_steps_per_trajectory:
                        for key in ['obs', 'actions']:
                            if key in episode_data:
                                episode_data[key] = episode_data[key][:max_steps_per_trajectory]
                    
                    # Collect states from this trajectory
                    trajectory_states = collect_trajectory_states_from_episode(episode_data, env_wrapper)
                    
                    # For now, create dummy coordinates - in joint training these will be real
                    n_states = len(trajectory_states)
                    dummy_coords = np.random.randn(n_states, 2) * 0.1 + checkpoint_idx
                    
                    # Add to collections
                    all_states.extend(trajectory_states)
                    all_coordinates.extend(dummy_coords)
                    checkpoint_indices.extend([checkpoint_idx] * n_states)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
                    
            logger.info(f"Collected {sum(1 for idx in checkpoint_indices if idx == checkpoint_idx)} states from checkpoint {checkpoint}")
    
    finally:
        env_wrapper.close()
    
    logger.info(f"Total states collected: {len(all_states)} from {len(checkpoints)} checkpoints")
    return all_states, np.array(all_coordinates), np.array(checkpoint_indices)
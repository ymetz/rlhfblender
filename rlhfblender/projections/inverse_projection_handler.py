import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from rlhfblender.projections.inverse_projection_architectures import get_inverse_model


class InverseProjectionHandler:
    """
    Handler for inverse projection tasks. Takes 2D coordinates from dimensionality
    reduction techniques and maps them back to the original high-dimensional space.
    """
    
    def __init__(self, model_type="auto", hidden_dims=None, learning_rate=0.001, 
                 batch_size=64, num_epochs=100, device=None, save_model=True, 
                 save_dir="models"):
        """
        Initialize the inverse projection handler.
        
        Args:
            model_type: Type of model to use ('auto', 'mlp', 'cnn', 'vae')
            hidden_dims: List of hidden dimensions for MLP and VAE models
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
            save_model: Whether to save the trained model
            save_dir: Directory to save the trained model
        """
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_model = save_model
        self.save_dir = save_dir
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.scaler = None  # For potential data scaling/normalization
        
        # Create save directory if it doesn't exist
        if self.save_model and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def fit(self, data, coords, validation_split=0.1, verbose=True):
        """
        Fit the inverse projection model to map from 2D coordinates to original data.
        
        Args:
            data: Original high-dimensional data (numpy array or torch tensor)
            coords: 2D coordinates from projection (numpy array or torch tensor)
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of training history (loss values per epoch)
        """
        # Convert data to PyTorch tensors if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        
        # Get data shape to determine the model architecture
        data_shape = data.shape
        
        # Initialize the model based on the data shape
        self.model = get_inverse_model(data_shape, self.model_type)
        if self.hidden_dims is not None and self.model_type in ["mlp", "vae"]:
            # Reinitialize with custom hidden dimensions if specified
            if self.model_type == "mlp":
                self.model = get_inverse_model(data_shape, "mlp", self.hidden_dims)
            elif self.model_type == "vae":
                self.model = get_inverse_model(data_shape, "vae", self.hidden_dims)
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Prepare data for training
        dataset = TensorDataset(coords, data)
        
        # Split into training and validation sets
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                        disable=not verbose)
            
            for batch_coords, batch_data in pbar:
                batch_coords, batch_data = batch_coords.to(self.device), batch_data.to(self.device)
                
                # Forward pass
                reconstructed = self.model(batch_coords)
                
                # Compute loss
                loss = self.loss_fn(reconstructed, batch_data)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_coords.size(0)
                pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_size > 0:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_coords, batch_data in val_loader:
                        batch_coords, batch_data = batch_coords.to(self.device), batch_data.to(self.device)
                        reconstructed = self.model(batch_coords)
                        loss = self.loss_fn(reconstructed, batch_data)
                        val_loss += loss.item() * batch_coords.size(0)
                
                val_loss /= len(val_dataset)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Save the model if requested
        if self.save_model:
            model_file = os.path.join(self.save_dir, f"inverse_projection_{self.model_type}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'data_shape': data_shape,
                'hidden_dims': self.hidden_dims
            }, model_file)
            if verbose:
                print(f"Model saved to {model_file}")
        
        return history
    
    def predict(self, coords):
        """
        Predict the original data points from 2D coordinates.
        
        Args:
            coords: 2D coordinates (numpy array or torch tensor)
            
        Returns:
            Predicted original data points
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Convert to PyTorch tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        
        # Move to device
        coords = coords.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict
        with torch.no_grad():
            predictions = self.model(coords)
        
        # Return as numpy array
        return predictions.cpu().numpy()
    
    def visualize_reconstructions(self, coords, original_data=None, num_samples=5, figsize=(15, 5)):
        """
        Visualize reconstructions from 2D coordinates.
        
        Args:
            coords: 2D coordinates
            original_data: Original data for comparison (optional)
            num_samples: Number of samples to visualize
            figsize: Figure size
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get reconstructions
        reconstructions = self.predict(coords[:num_samples])
        
        # Determine if we're dealing with images
        is_image = len(reconstructions.shape) > 2
        
        # Create figure
        if is_image:
            fig, axes = plt.subplots(2 if original_data is not None else 1, 
                                    num_samples, figsize=figsize)
            
            for i in range(num_samples):
                # Plot reconstruction
                recon_img = reconstructions[i]
                # Transpose if needed (channels last to channels first)
                if recon_img.shape[0] not in [1, 3]:
                    recon_img = np.transpose(recon_img, (1, 2, 0))
                
                if original_data is not None:
                    axes[1, i].imshow(recon_img)
                    axes[1, i].set_title(f"Reconstruction {i+1}")
                    axes[1, i].axis('off')
                    
                    # Plot original
                    orig_img = original_data[i]
                    if orig_img.shape[0] not in [1, 3]:
                        orig_img = np.transpose(orig_img, (1, 2, 0))
                    axes[0, i].imshow(orig_img)
                    axes[0, i].set_title(f"Original {i+1}")
                    axes[0, i].axis('off')
                else:
                    axes[i].imshow(recon_img)
                    axes[i].set_title(f"Reconstruction {i+1}")
                    axes[i].axis('off')
        else:
            # For non-image data, create a simple plot of feature values
            fig, axes = plt.subplots(2 if original_data is not None else 1, 
                                    1, figsize=figsize)
            
            for i in range(num_samples):
                # Plot reconstruction
                if original_data is not None:
                    axes[1].plot(reconstructions[i], label=f"Reconstruction {i+1}")
                    # Plot original
                    axes[0].plot(original_data[i], label=f"Original {i+1}")
                else:
                    axes.plot(reconstructions[i], label=f"Reconstruction {i+1}")
            
            if original_data is not None:
                axes[0].set_title("Original Data")
                axes[0].legend()
                axes[1].set_title("Reconstructions")
                axes[1].legend()
            else:
                axes.set_title("Reconstructions")
                axes.legend()
        
        plt.tight_layout()
        plt.show()
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model information
        model_type = checkpoint.get('model_type', 'mlp')
        data_shape = checkpoint.get('data_shape')
        hidden_dims = checkpoint.get('hidden_dims')
        
        # Initialize the model
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.model = get_inverse_model(data_shape, model_type)
        
        # Load the state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def create_latent_space_grid(self, x_range=(-5, 5), y_range=(-5, 5), 
                                resolution=20, return_coords=False):
        """
        Create a grid in the latent space and generate reconstructions for visualization.
        
        Args:
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Number of points per dimension
            return_coords: Whether to return the grid coordinates
            
        Returns:
            Grid of reconstructions and optionally grid coordinates
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        
        # Flatten grid to list of coordinates
        coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
        
        # Generate reconstructions
        reconstructions = self.predict(coords)
        
        if return_coords:
            return reconstructions, coords, (grid_x, grid_y)
        return reconstructions


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    
    # 1D vector example
    data_dim = 30
    num_samples = 1000
    
    # Generate random data
    original_data = np.random.randn(num_samples, data_dim)
    
    # Generate random 2D coordinates (simulating projection output)
    coords_2d = np.random.randn(num_samples, 2)
    
    # Create and fit the handler
    handler = InverseProjectionHandler(
        model_type="mlp",
        num_epochs=20,
        batch_size=32,
        verbose=True
    )
    
    history = handler.fit(original_data, coords_2d)
    
    # Generate predictions
    test_coords = np.random.randn(5, 2)
    predictions = handler.predict(test_coords)
    
    print(f"Input shape: {test_coords.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Visualize reconstructions
    handler.visualize_reconstructions(coords_2d, original_data)
    
    # Create and visualize a latent space grid
    grid_recon = handler.create_latent_space_grid(resolution=10)
    print(f"Grid shape: {grid_recon.shape}")
import base64
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rlhfblender.projections.inverse_projection_architectures import get_inverse_model


class InverseProjectionHandler:
    """
    Handler for inverse projection tasks. Takes 2D coordinates from dimensionality
    reduction techniques and maps them back to the original high-dimensional space.
    """

    def __init__(
        self,
        model_type="auto",
        hidden_dims=None,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=30,
        device=None,
        save_model=True,
        save_dir="models",
        weight_decay=0.01,
        use_scheduler=True,
        warmup_epochs=3,
        min_lr_ratio=0.1,
    ):
        """
        Initialize the inverse projection handler.

        Args:
            model_type: Type of model to use ('auto', 'mlp', 'cnn', 'vae')
            hidden_dims: List of hidden dimensions for MLP and VAE models
            learning_rate: Initial learning rate for the optimizer
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
            save_model: Whether to save the trained model
            save_dir: Directory to save the trained model
            weight_decay: Weight decay for AdamW optimizer
            use_scheduler: Whether to use learning rate scheduler
            warmup_epochs: Number of epochs for linear warmup
            min_lr_ratio: Minimum learning rate as ratio of initial learning rate
        """
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_model = save_model
        self.save_dir = save_dir
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.MSELoss()
        self.scaler = None  # For potential data scaling/normalization

        # Create save directory if it doesn't exist
        if self.save_model and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _create_scheduler(self, total_steps):
        """
        Create a learning rate scheduler with linear warmup followed by linear decay.

        Args:
            total_steps: Total number of training steps
        """
        if not self.use_scheduler:
            return None

        warmup_steps = self.warmup_epochs * (total_steps // self.num_epochs)
        decay_steps = total_steps - warmup_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Linear decay
                progress = (step - warmup_steps) / decay_steps
                return max(self.min_lr_ratio, 1.0 - progress * (1.0 - self.min_lr_ratio))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def fit(self, data, coords, validation_split=0.1, verbose=True):
        """
        Fit the inverse projection model to map from 2D coordinates to original data.

        Args:
            data: Original high-dimensional data (numpy array or torch tensor)
            coords: 2D coordinates from projection (numpy array or torch tensor)
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress

        Returns:
            Dictionary of training history (loss values per epoch and learning rates)
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

        # Initialize AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Prepare data for training
        dataset = TensorDataset(coords, data)

        # Split into training and validation sets
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.num_epochs

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(total_steps)

        # Training history
        history = {"train_loss": [], "val_loss": [], "learning_rates": []}

        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            epoch_lrs = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", disable=not verbose)

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

                # Update learning rate scheduler
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    epoch_lrs.append(current_lr)
                    self.scheduler.step()
                else:
                    epoch_lrs.append(self.learning_rate)

                train_loss += loss.item() * batch_coords.size(0)

                # Update progress bar with current learning rate
                current_lr = epoch_lrs[-1] if epoch_lrs else self.learning_rate
                pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.2e}"})

            train_loss /= len(train_dataset)
            history["train_loss"].append(train_loss)
            history["learning_rates"].append(np.mean(epoch_lrs))

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
                history["val_loss"].append(val_loss)

                if verbose:
                    current_lr = history["learning_rates"][-1]
                    print(
                        f"Epoch {epoch+1}/{self.num_epochs}, "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
            else:
                if verbose:
                    current_lr = history["learning_rates"][-1]
                    print(f"Epoch {epoch+1}/{self.num_epochs}, " f"Train Loss: {train_loss:.4f}, " f"LR: {current_lr:.2e}")

        # Save the model if requested
        if self.save_model:
            model_file = os.path.join(self.save_dir, f"inverse_projection_{self.model_type}.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_type": self.model_type,
                    "data_shape": data_shape,
                    "hidden_dims": self.hidden_dims,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "weight_decay": self.weight_decay,
                    "use_scheduler": self.use_scheduler,
                    "warmup_epochs": self.warmup_epochs,
                    "min_lr_ratio": self.min_lr_ratio,
                },
                model_file,
            )
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
            fig, axes = plt.subplots(2 if original_data is not None else 1, num_samples, figsize=figsize)

            for i in range(num_samples):
                # Plot reconstruction
                recon_img = reconstructions[i]
                # Transpose if needed (channels last to channels first)
                if recon_img.shape[0] not in [1, 3]:
                    recon_img = np.transpose(recon_img, (1, 2, 0))

                if original_data is not None:
                    axes[1, i].imshow(recon_img)
                    axes[1, i].set_title(f"Reconstruction {i+1}")
                    axes[1, i].axis("off")

                    # Plot original
                    orig_img = original_data[i]
                    if orig_img.shape[0] not in [1, 3]:
                        orig_img = np.transpose(orig_img, (1, 2, 0))
                    axes[0, i].imshow(orig_img)
                    axes[0, i].set_title(f"Original {i+1}")
                    axes[0, i].axis("off")
                else:
                    axes[i].imshow(recon_img)
                    axes[i].set_title(f"Reconstruction {i+1}")
                    axes[i].axis("off")
        else:
            # For non-image data, create a simple plot of feature values
            fig, axes = plt.subplots(2 if original_data is not None else 1, 1, figsize=figsize)

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
        model_type = checkpoint.get("model_type", "mlp")
        data_shape = checkpoint.get("data_shape")
        hidden_dims = checkpoint.get("hidden_dims")

        # Initialize the model
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.model = get_inverse_model(data_shape, model_type)

        # Load the state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def create_latent_space_grid(self, x_range=(-5, 5), y_range=(-5, 5), resolution=20, return_coords=False):
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

    @staticmethod
    def precompute_interpolated_surface(
        grid_coords,
        grid_values,
        additional_coords=None,
        additional_values=None,
        resolution=400,
        method="linear",
        mask_radius=None,
    ):
        """
        Creates an interpolated surface prioritizing original data over grid data.
        """
        # Convert to numpy arrays
        grid_coords = np.array(grid_coords)
        grid_values = np.array(grid_values)

        # If we have original data points to incorporate
        if additional_coords is not None and additional_values is not None:
            additional_coords = np.array(additional_coords)
            additional_values = np.array(additional_values)

            # Calculate optimal mask_radius if not provided
            if mask_radius is None:
                # Compute average distance between neighboring grid points
                from scipy.spatial import KDTree

                tree = KDTree(grid_coords)
                distances, _ = tree.query(grid_coords, k=5)  # query k=5 nearest neighbors
                avg_dist = np.mean(distances[:, 1:])  # exclude self-distance (first column)
                mask_radius = avg_dist * 0.75  # 75% of average distance is a good default

            # Create a mask for grid points that should be excluded
            grid_mask = np.ones(len(grid_coords), dtype=bool)

            # For each original point, mask nearby grid points
            for orig_coord in additional_coords:
                distances = np.sqrt(np.sum((grid_coords - orig_coord) ** 2, axis=1))
                grid_mask = grid_mask & (distances >= mask_radius)

            # Apply the mask to keep only grid points that are far from original points
            masked_grid_coords = grid_coords[grid_mask]
            masked_grid_values = grid_values[grid_mask]

            # Combine masked grid points with original points
            points = np.vstack((masked_grid_coords, additional_coords))
            values = np.append(masked_grid_values, additional_values)
        else:
            # If no original points, use all grid points
            points = grid_coords
            values = grid_values

        # Rest of function remains the same...
        x_min, x_max = min(points[:, 0]), max(points[:, 0])
        y_min, y_max = min(points[:, 1]), max(points[:, 1])

        # Add a small buffer to avoid edge effects
        x_buffer = 0
        y_buffer = 0

        xi = np.linspace(x_min - x_buffer, x_max + x_buffer, resolution)
        yi = np.linspace(y_min - y_buffer, y_max + y_buffer, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Perform the interpolation
        if method == "rbf":
            rbf = Rbf(points[:, 0], points[:, 1], values, function="multiquadric")
            zi_grid = rbf(xi_grid, yi_grid)
        else:
            zi_grid = griddata(points, values, (xi_grid, yi_grid), method=method)

            # Fill NaN values that might occur at the edges
            if np.any(np.isnan(zi_grid)):
                zi_grid_nearest = griddata(points, values, (xi_grid, yi_grid), method="nearest")
                zi_grid = np.where(np.isnan(zi_grid), zi_grid_nearest, zi_grid)

        zi_grid = griddata(points, values, (xi_grid, yi_grid), method="linear")
        zi_grid = gaussian_filter(zi_grid, sigma=1.5)

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8), dpi=resolution / 8)
        norm = Normalize(vmin=min(values), vmax=max(values))
        im = ax.pcolormesh(xi_grid, yi_grid, zi_grid, shading="auto", cmap="viridis", norm=norm)

        # Optionally mark original data points
        if additional_coords is not None:
            pass
            # ax.scatter(additional_coords[:, 0], additional_coords[:, 1], s=5, c='white', alpha=0.5, marker='o')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        plt.tight_layout()
        plt.axis("off")

        # Save to base64-encoded PNG
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # Return metadata along with the image
        return {
            "image": img_str,
            "min_value": float(min(values)),
            "max_value": float(max(values)),
            "x_range": [float(x_min) - x_buffer, float(x_max) + x_buffer],
            "y_range": [float(y_min) - y_buffer, float(y_max) + y_buffer],
            "point_count": len(points),
            "interpolation_method": method,
            "mask_radius": float(mask_radius) if mask_radius is not None else None,
            "original_points_count": len(additional_coords) if additional_coords is not None else 0,
            "grid_points_used": len(masked_grid_coords) if additional_coords is not None else len(grid_coords),
        }

    @staticmethod
    def interpolate_brbg(t):
        """
        Python implementation of d3.interpolateBrBG color scheme.
        
        Args:
            t: Value between 0 and 1
            
        Returns:
            RGB color as numpy array [r, g, b] in [0, 1] range
        """
        # BrBG color scheme (Brown-Blue-Green) color stops
        # Based on d3.interpolateBrBG
        colors = [
            [0.329, 0.188, 0.020],  # Dark brown
            [0.549, 0.318, 0.039],  # Brown
            [0.749, 0.506, 0.176],  # Light brown
            [0.874, 0.761, 0.490],  # Tan
            [0.964, 0.909, 0.764],  # Light tan
            [0.961, 0.961, 0.961],  # White (center)
            [0.780, 0.917, 0.898],  # Light blue-green
            [0.502, 0.804, 0.757],  # Blue-green
            [0.207, 0.592, 0.561],  # Teal
            [0.004, 0.424, 0.376],  # Dark teal
            [0.000, 0.235, 0.188],  # Very dark teal
        ]
        
        colors = np.array(colors)
        
        # Clamp t to [0, 1]
        t = np.clip(t, 0, 1)
        
        # Scale t to color array indices
        scaled_t = t * (len(colors) - 1)
        
        # Find the two colors to interpolate between
        idx = int(np.floor(scaled_t))
        idx = min(idx, len(colors) - 2)  # Ensure we don't go out of bounds
        
        # Interpolation factor
        frac = scaled_t - idx
        
        # Linear interpolation between the two colors
        color1 = colors[idx]
        color2 = colors[idx + 1]
        
        return color1 + frac * (color2 - color1)

    @staticmethod
    def generate_vsup_colormap(resolution=256, num_bins=5):
        """
        Generate a Value-Suppressing Uncertainty Palette matching frontend VSUP logic:
        - Uses BrBG color scheme for prediction values
        - Uses interpolation to white for uncertainty (matching VSUP "usl" mode)
        - uncertainty=0 -> full color, uncertainty=1 -> white
        - Optional quantization into discrete bins

        Parameters:
        -----------
        resolution: Resolution of the colormap
        num_bins: Number of discrete bins (set to 0 for continuous map)

        Returns:
        --------
        colormap: (resolution x resolution x 3) RGB values in [0, 1]
        """
        # Create grid
        grid = np.zeros((resolution, resolution, 3))

        # Apply discretization if requested
        if num_bins > 1:
            # Create bin edges for value and uncertainty
            value_bins = np.linspace(0, 1, num_bins + 1)
            uncertainty_bins = np.linspace(0, 1, num_bins + 1)

            # For each bin combination, calculate appropriate color
            for i in range(num_bins):
                for j in range(num_bins):
                    # Bin centers
                    v_center = (value_bins[j] + value_bins[j + 1]) / 2
                    u_center = (uncertainty_bins[i] + uncertainty_bins[i + 1]) / 2

                    # Get base color from BrBG scheme based on prediction
                    base_color = InverseProjectionHandler.interpolate_brbg(v_center)

                    # Apply uncertainty effect - interpolate with white
                    # uncertainty=0 -> full color, uncertainty=1 -> white
                    white = np.array([1.0, 1.0, 1.0])
                    adjusted_color = (1 - u_center) * base_color + u_center * white

                    # Fill the bin area with this color
                    v_min, v_max = int(value_bins[j] * resolution), int(value_bins[j + 1] * resolution)
                    u_min, u_max = int(uncertainty_bins[i] * resolution), int(uncertainty_bins[i + 1] * resolution)

                    # Handle edge case for last bin
                    if j == num_bins - 1:
                        v_max = resolution
                    if i == num_bins - 1:
                        u_max = resolution

                    grid[u_min:u_max, v_min:v_max] = adjusted_color
        else:
            # Continuous version (no bins)
            for i in range(resolution):
                u = i / (resolution - 1)  # uncertainty (0-1)
                for j in range(resolution):
                    v = j / (resolution - 1)  # value (0-1)

                    # Get base color from BrBG scheme based on prediction
                    base_color = InverseProjectionHandler.interpolate_brbg(v)

                    # Apply uncertainty effect - interpolate with white
                    # This matches VSUP's: d3.interpolateLab(vcolor, "#fff")(uScale(data.u))
                    # uncertainty=0 -> full color, uncertainty=1 -> white
                    white = np.array([1.0, 1.0, 1.0])
                    adjusted_color = (1 - u) * base_color + u * white

                    grid[i, j] = adjusted_color

        return grid

    @staticmethod
    def precompute_bivariate_interpolated_surface(
        grid_coords,
        grid_predictions,
        grid_uncertainties,
        additional_coords=None,
        additional_predictions=None,
        additional_uncertainties=None,
        resolution=400,
        method="linear",
        mask_radius=None,
    ):
        """
        Creates a bivariate interpolated surface combining prediction and uncertainty,
        prioritizing original data over grid data.
        """
        # Convert to numpy arrays
        grid_coords = np.array(grid_coords)
        grid_predictions = np.array(grid_predictions)
        grid_uncertainties = np.array(grid_uncertainties)

        # If we have original data points
        if additional_coords is not None and additional_predictions is not None and additional_uncertainties is not None:
            additional_coords = np.array(additional_coords)
            additional_predictions = np.array(additional_predictions)
            additional_uncertainties = np.array(additional_uncertainties)

            # Calculate optimal mask_radius if not provided
            if mask_radius is None:
                # Compute average distance between neighboring grid points
                from scipy.spatial import KDTree

                tree = KDTree(grid_coords)
                distances, _ = tree.query(grid_coords, k=5)  # query k=5 nearest neighbors
                avg_dist = np.mean(distances[:, 1:])  # exclude self-distance
                mask_radius = avg_dist * 0.75  # 75% of average distance is a good default

            # Create a mask for grid points that should be excluded
            grid_mask = np.ones(len(grid_coords), dtype=bool)

            # For each original point, mask nearby grid points
            for orig_coord in additional_coords:
                distances = np.sqrt(np.sum((grid_coords - orig_coord) ** 2, axis=1))
                grid_mask = grid_mask & (distances >= mask_radius)

            # Apply the mask to keep only grid points that are far from original points
            masked_grid_coords = grid_coords[grid_mask]
            masked_grid_predictions = grid_predictions[grid_mask]
            masked_grid_uncertainties = grid_uncertainties[grid_mask]

            # Combine masked grid points with original points
            coords = np.vstack((masked_grid_coords, additional_coords))
            preds = np.append(masked_grid_predictions, additional_predictions)
            uncertainties = np.append(masked_grid_uncertainties, additional_uncertainties)
        else:
            # If no original points, use all grid points
            coords = grid_coords
            preds = grid_predictions
            uncertainties = grid_uncertainties

        # Rest of function remains largely the same...
        x_min, x_max = min(coords[:, 0]), max(coords[:, 0])
        y_min, y_max = min(coords[:, 1]), max(coords[:, 1])

        x_buffer = 0
        y_buffer = 0

        xi = np.linspace(x_min - x_buffer, x_max + x_buffer, resolution)
        yi = np.linspace(y_min - y_buffer, y_max + y_buffer, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Perform the interpolation
        zi_pred = griddata(coords, preds, (xi_grid, yi_grid), method=method)
        zi_uncertainty = griddata(coords, uncertainties, (xi_grid, yi_grid), method=method)

        zi_pred = gaussian_filter(zi_pred, sigma=1.5)
        zi_uncertainty = gaussian_filter(zi_uncertainty, sigma=1.5)

        # Fill NaN values
        if np.any(np.isnan(zi_pred)):
            zi_pred_nearest = griddata(coords, preds, (xi_grid, yi_grid), method="nearest")
            zi_pred = np.where(np.isnan(zi_pred), zi_pred_nearest, zi_pred)

        if np.any(np.isnan(zi_uncertainty)):
            zi_uncertainty_nearest = griddata(coords, uncertainties, (xi_grid, yi_grid), method="nearest")
            zi_uncertainty = np.where(np.isnan(zi_uncertainty), zi_uncertainty_nearest, zi_uncertainty)

        # Create a color-mapped image
        fig, ax = plt.subplots(figsize=(8, 8), dpi=resolution / 8)

        # Draw the interpolated surface
        norm_pred = Normalize(vmin=min(preds), vmax=max(preds))
        norm_uncertainty = Normalize(vmin=min(uncertainties), vmax=max(uncertainties))

        zi_pred_norm = norm_pred(zi_pred)
        zi_uncertainty_norm = norm_uncertainty(zi_uncertainty)

        # Apply bivariate colormap
        bimap = InverseProjectionHandler.generate_vsup_colormap(resolution=resolution)
        zi_pred_norm = norm_pred(zi_pred)
        zi_uncertainty_norm = norm_uncertainty(zi_uncertainty)

        zi_pred_norm = np.clip(zi_pred_norm, 0, 1)
        zi_uncertainty_norm = np.clip(zi_uncertainty_norm, 0, 1)

        idx_x = (zi_pred_norm * (resolution - 1)).astype(int)
        idx_y = (zi_uncertainty_norm * (resolution - 1)).astype(int)
        rgb = bimap[idx_y, idx_x]

        ax.imshow(
            rgb,
            origin="lower",
            extent=[x_min - x_buffer, x_max + x_buffer, y_min - y_buffer, y_max + y_buffer],
            interpolation="bilinear",
        )

        # Optionally mark original data points
        if additional_coords is not None:
            pass
            # ax.scatter(additional_coords[:, 0], additional_coords[:, 1], s=5, c='white', alpha=0.5, marker='o')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        plt.tight_layout()
        plt.axis("off")

        # Save to base64-encoded PNG
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # Return metadata with additional information
        return {
            "image": img_str,
            "min_value": float(min(preds)),
            "max_value": float(max(preds)),
            "x_range": [float(x_min) - x_buffer, float(x_max) + x_buffer],
            "y_range": [float(y_min) - y_buffer, float(y_max) + y_buffer],
            "point_count": len(coords),
            "interpolation_method": method,
            "mask_radius": float(mask_radius) if mask_radius is not None else None,
            "original_points_count": len(additional_coords) if additional_coords is not None else 0,
            "grid_points_used": len(masked_grid_coords) if additional_coords is not None else len(grid_coords),
        }


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
    handler = InverseProjectionHandler(model_type="mlp", num_epochs=20, batch_size=32, verbose=True)

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

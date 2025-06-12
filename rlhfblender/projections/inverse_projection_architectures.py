import numpy as np
import torch
import torch.nn as nn


class MLPInverseModel(nn.Module):
    """
    MLP-based inverse projection model for 1D vector data.
    Maps from 2D latent coordinates back to original high-dimensional space.
    """

    def __init__(self, output_dim, hidden_dims=[480, 640, 1024]):
        super().__init__()

        # Start with input dimension of 2 (2D coordinates)
        layers = []
        input_dim = 2

        # Build the network with the specified hidden dimensions
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.25))
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CNNInverseModel(nn.Module):
    """
    CNN-based inverse projection model for image data.
    Maps from 2D latent coordinates to image reconstruction.
    """

    def __init__(self, output_shape):
        super().__init__()

        # Store original output shape
        self.output_shape = output_shape

        # Calculate final output dimension
        final_output_dim = np.prod(output_shape)

        # Initial fully connected layers to increase dimensionality
        self.fc_layers = nn.Sequential(
            nn.Linear(2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.starting_width = max(4, output_shape[1] // 4)
        self.starting_height = max(4, output_shape[2] // 4)

        # Adjust the fc_to_conv layer accordingly
        self.fc_to_conv = nn.Linear(2048, 128 * self.starting_width * self.starting_height)

        # Transposed convolution with dynamic structure
        layers = []
        in_channels = 128
        current_width = self.starting_width
        current_height = self.starting_height

        # Add layers until we reach the target size
        while current_width < output_shape[1] or current_height < output_shape[2]:
            out_channels = max(in_channels // 2, output_shape[0])
            layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )

            in_channels = out_channels
            current_width *= 2
            current_height *= 2

        # Final layer to get to the correct number of channels
        layers.append(nn.ConvTranspose2d(in_channels, output_shape[0], kernel_size=3, stride=1, padding=1))

        self.conv_transpose_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Pass through fully connected layers
        x = self.fc_layers(x)

        # Reshape for convolutional layers
        x = self.fc_to_conv(x)
        x = x.view(-1, 256, self.starting_width, self.starting_width)

        # Pass through transposed convolutional layers
        x = self.conv_transpose_layers(x)

        # Ensure output has the correct shape
        batch_size = x.shape[0]
        return x.view(batch_size, *self.output_shape)


class VAEInverseModel(nn.Module):
    """
    Variational Autoencoder-based inverse projection model.
    Maps from 2D latent coordinates back to original high-dimensional space,
    with the ability to sample from the learned distribution.
    """

    def __init__(self, output_dim, hidden_dims=[480, 640, 1024]):
        super().__init__()

        # Decoder network (similar to MLP model)
        layers = []
        input_dim = 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.25))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Standard forward pass
        return self.decode(x)


def get_inverse_model(data_shape, model_type="auto", hidden_dims=[64, 128, 192]):
    """
    Factory function to create the appropriate inverse model based on the data shape.

    Args:
        data_shape: Tuple representing the shape of the original data
        model_type: The type of model to use ("mlp", "cnn", "vae")

    Returns:
        An instance of the appropriate inverse model
    """
    if len(data_shape) > 2:  # Image data (batch_size, channels, height, width)
        if model_type == "cnn" or model_type == "auto":
            return CNNInverseModel(data_shape[1:])
        elif model_type == "vae":
            # Flatten the dimensions for the VAE
            output_dim = np.prod(data_shape[1:])
            model = VAEInverseModel(output_dim, hidden_dims=hidden_dims)
            # Add method to reshape the output back to the original image shape
            original_forward = model.forward

            def new_forward(x):
                flat_output = original_forward(x)
                return flat_output.view(-1, *data_shape[1:])

            model.forward = new_forward
            return model

    # 1D vector data (batch_size, features) or flattened image data
    output_dim = data_shape[1] if len(data_shape) > 1 else data_shape[0]

    if model_type == "vae":
        return VAEInverseModel(output_dim)
    else:  # Default to MLP
        return MLPInverseModel(output_dim)

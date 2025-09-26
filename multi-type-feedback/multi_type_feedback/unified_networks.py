from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from masksembles.torch import Masksembles1D, Masksembles2D
from pytorch_lightning import LightningModule
from torch import Tensor, nn


class UnifiedNetwork(LightningModule):
    """
    A unified network that handles all feedback types with a single model.
    Uses feedback type embeddings to differentiate between feedback types.
    """

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int = 6,
        output_dim: int = 1,
        hidden_dim: int = 256,
        action_hidden_dim: int = 32,
        feedback_types: List[str] = None,
        learning_rate: float = 1e-5,
        cnn_channels: List[int] = None,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 4,
        masksemble_scale: float = 1.8,
        feedback_embedding_dim: int = 32,
    ):
        super().__init__()

        if feedback_types is None:
            feedback_types = [
                "evaluative",
                "comparative",
                "demonstrative",
                "descriptive",
                "descriptive_preference",
                "corrective",
                "supervised",
            ]

        self.feedback_types = feedback_types
        self.feedback_type_map = {
            fb_type: i for i, fb_type in enumerate(feedback_types)
        }
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        self.masksemble_scale = masksemble_scale

        obs_space, action_space = input_spaces
        action_is_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Determine input dimension
        input_dim = np.prod(obs_space.shape) + (
            np.prod(action_space.shape) if not action_is_discrete else action_space.n
        )

        # Create feedback type embedding
        self.feedback_embedding = nn.Embedding(
            len(feedback_types), feedback_embedding_dim
        )

        # Adjust input dimension to include feedback embedding
        augmented_input_dim = input_dim + feedback_embedding_dim

        # Create network layers
        layers = []
        current_dim = augmented_input_dim

        for _ in range(layer_num - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_function())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=hidden_dim,
                        n=self.ensemble_count,
                        scale=self.masksemble_scale,
                    ).float()
                )

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        if last_activation is not None:
            layers.append(last_activation())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=output_dim,
                        n=self.ensemble_count,
                        scale=self.masksemble_scale,
                    ).float()
                )

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        # Define loss normalization parameters (learned)
        self.loss_scale = nn.Parameter(torch.ones(len(feedback_types)))
        self.loss_bias = nn.Parameter(torch.zeros(len(feedback_types)))

        self.save_hyperparameters()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, observations: Tensor, actions: Tensor, feedback_type: str):
        """
        Forward pass through the network with feedback type conditioning.
        """

        if len(observations.shape) > 3:  # For 2D spaces like highway-env
            observations = observations.flatten(start_dim=2)

        batch_size, segment_length, obs_dim = observations.shape
        _, _, action_dim = actions.shape

        # Get feedback type index and embedding
        feedback_idx = self.feedback_type_map[feedback_type]
        feedback_embedding = self.feedback_embedding(
            torch.tensor(feedback_idx, device=observations.device)
        )

        # Expand embedding to match batch and sequence dimensions
        # Shape: (batch_size, segment_length, embedding_dim)
        expanded_embedding = feedback_embedding.expand(batch_size, segment_length, -1)

        # Flatten the batch and sequence dimensions
        obs_flat = observations.reshape(-1, obs_dim)
        actions_flat = actions.reshape(-1, action_dim)
        embedding_flat = expanded_embedding.reshape(-1, feedback_embedding.shape[-1])

        # Concatenate observations, actions, and feedback embedding
        x = torch.cat((obs_flat, actions_flat, embedding_flat), dim=1)

        # Pass through network
        output = self.network(x)

        # Reshape back
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def universal_loss(self, batch: Tensor):
        """
        A universal loss function that handles all feedback types with optimized ensemble handling.
        """
        feedback_type, data = batch

        if isinstance(feedback_type, (list, tuple, torch.Tensor)):
            feedback_type = feedback_type[0]
        if isinstance(feedback_type, torch.Tensor):
            feedback_type = feedback_type.item()
    
        feedback_idx = self.feedback_type_map[feedback_type]

        # Skip ensemble repetition if ensemble_count is 1
        if self.ensemble_count <= 1:
            # Handle regular (non-ensemble) case directly
            # [existing code without repetition]
            pass

        if feedback_type in [
            "comparative",
            "demonstrative",
            "corrective",
            "descriptive_preference",
        ]:
            # Handle pairwise comparison feedback
            pair_data, preferred_indices = data
            (obs1, actions1, mask1), (obs2, actions2, mask2) = pair_data

            # For ensemble models, use optimized batch repetition
            if self.ensemble_count > 1:
                # Create repeat pattern once - for ensembles we only expand the batch dimension
                obs_repeat = [self.ensemble_count] + [1] * (len(obs1.shape) - 1)
                act_repeat = [self.ensemble_count] + [1] * (len(actions1.shape) - 1)
                mask_repeat = [self.ensemble_count] + [1] * (len(mask1.shape) - 1)

                # Batch the repetition operations (helps compiler optimize)
                obs1 = obs1.repeat(*obs_repeat)
                actions1 = actions1.repeat(*act_repeat)
                mask1 = mask1.repeat(*mask_repeat)
                obs2 = obs2.repeat(*obs_repeat)
                actions2 = actions2.repeat(*act_repeat)
                mask2 = mask2.repeat(*mask_repeat)
                preferred_indices = preferred_indices.repeat(
                    self.ensemble_count, 1
                ).squeeze()
                print("PREFERRED INDICES", obs1.shape, preferred_indices, preferred_indices.shape)

            # Compute network outputs for both trajectories
            outputs1 = self.forward(obs1, actions1, feedback_type)
            outputs2 = self.forward(obs2, actions2, feedback_type)

            # Sum over sequence dimension with masking
            rewards1 = (outputs1 * mask1).sum(dim=1).squeeze(-1)
            rewards2 = (outputs2 * mask2).sum(dim=1).squeeze(-1)

            # Apply learned normalization
            scale = torch.abs(self.loss_scale[feedback_idx]) + 1e-6  # Ensure positive
            bias = self.loss_bias[feedback_idx]

            normalized_rewards1 = rewards1 * scale + bias
            normalized_rewards2 = rewards2 * scale + bias

            # Stack rewards and compute log softmax
            rewards = torch.stack([normalized_rewards1, normalized_rewards2], dim=1)
            log_probs = F.log_softmax(rewards, dim=1)

            # Compute NLL loss
            print("LOG PROBS AND PREF. INDICES", log_probs, preferred_indices)
            loss = F.nll_loss(log_probs, preferred_indices)

        elif feedback_type in ["evaluative", "descriptive", "supervised"]:
            # Handle scalar feedback
            (observations, actions, masks), targets = data

            # For ensemble models, use optimized batch repetition
            if self.ensemble_count > 1:
                # Create repeat pattern once
                obs_repeat = [self.ensemble_count] + [1] * (len(observations.shape) - 1)
                act_repeat = [self.ensemble_count] + [1] * (len(actions.shape) - 1)
                mask_repeat = [self.ensemble_count] + [1] * (len(masks.shape) - 1)

                # Batch repetition operations - compiler can optimize these better
                observations = observations.repeat(*obs_repeat)
                actions = actions.repeat(*act_repeat)
                masks = masks.repeat(*mask_repeat)

                # Convert targets to float and repeat
                targets = targets.float().repeat(self.ensemble_count, 1).squeeze()
                print("targets", observations, observations.shape, targets, targets.shape)

            # Network output: (batch_size, segment_length, output_dim)
            outputs = self.forward(observations, actions, feedback_type)

            # Sum over the sequence dimension to get total rewards per segment
            total_rewards = (outputs * masks).sum(dim=1).squeeze(-1)

            # Apply learned normalization
            scale = torch.abs(self.loss_scale[feedback_idx]) + 1e-6  # Ensure positive
            bias = self.loss_bias[feedback_idx]

            normalized_rewards = total_rewards * scale + bias

            # Compute MSE loss
            print("LOG PROBS AND PREF. INDICES", normalized_rewards, targets)
            loss = F.mse_loss(normalized_rewards, targets)

        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

        return loss

    def training_step(self, batch: Tensor, batch_idx: int):
        """Compute universal loss for training."""
        loss = self.universal_loss(batch)

        feedback_type = batch[0]
        if isinstance(feedback_type, (list, tuple, torch.Tensor)):
            feedback_type = feedback_type[0]
        if isinstance(feedback_type, torch.Tensor):
            feedback_type = feedback_type.item()

        self.log(f"train_loss_{feedback_type}", loss, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)

        # Also log the learned normalization parameters
        if feedback_type in self.feedback_type_map:
            idx = self.feedback_type_map[feedback_type]
            self.log(f"norm_scale_{feedback_type}", self.loss_scale[idx], on_epoch=True)
            self.log(f"norm_bias_{feedback_type}", self.loss_bias[idx], on_epoch=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """Compute universal loss for validation."""
        loss = self.universal_loss(batch)

        feedback_type = batch[0]
        self.log(f"val_loss_{feedback_type}", loss, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class UnifiedCnnNetwork(LightningModule):
    """
    A unified CNN network that handles all feedback types with a single model.
    Uses feedback type embeddings to differentiate between feedback types.
    """

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 256,
        action_hidden_dim: int = 16,
        feedback_types: List[str] = None,
        learning_rate: float = 1e-5,
        cnn_channels: List[int] = (16, 32, 32),
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 4,
        masksemble_scale: float = 1.8,
        feedback_embedding_dim: int = 32,
    ):
        super().__init__()

        if feedback_types is None:
            feedback_types = [
                "evaluative",
                "comparative",
                "demonstrative",
                "descriptive",
                "descriptive_preference",
                "corrective",
                "supervised",
            ]

        self.feedback_types = feedback_types
        self.feedback_type_map = {
            fb_type: i for i, fb_type in enumerate(feedback_types)
        }
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        self.masksemble_scale = masksemble_scale

        obs_space, action_space = input_spaces
        input_channels = obs_space.shape[0]

        # Create CNN layers
        cnn_layers = []
        for i in range(min(layer_num, len(cnn_channels))):
            cnn_layers.append(
                self.conv_sequence(
                    input_channels if i == 0 else cnn_channels[i - 1], cnn_channels[i]
                )
            )

        self.conv_layers = nn.Sequential(*cnn_layers)
        self.flatten = nn.Flatten()

        # Create feedback type embedding
        self.feedback_embedding = nn.Embedding(
            len(feedback_types), feedback_embedding_dim
        )

        # Action input layer
        action_shape = action_space.shape if action_space.shape else 1
        self.action_in = nn.Linear(action_shape, action_hidden_dim)
        self.action_masksemble = Masksembles1D(
            channels=action_hidden_dim,
            n=self.ensemble_count,
            scale=self.masksemble_scale,
        ).float()

        # Calculate CNN output size and combine with action and feedback embedding
        self.cnn_out_size = self.compute_flattened_size(
            obs_space.shape, cnn_channels[:layer_num]
        )
        combined_size = self.cnn_out_size + action_hidden_dim + feedback_embedding_dim

        # Final fully connected layer
        self.fc = nn.Linear(combined_size, output_dim)

        # Define loss normalization parameters (learned)
        self.loss_scale = nn.Parameter(torch.ones(len(feedback_types)))
        self.loss_bias = nn.Parameter(torch.zeros(len(feedback_types)))

        self.save_hyperparameters()

    def conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def residual_block(self, in_channels):
        return nn.Sequential(
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=self.masksemble_scale
            ).float(),
            self.conv_layer(in_channels, in_channels),
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=self.masksemble_scale
            ).float(),
            self.conv_layer(in_channels, in_channels),
        )

    def conv_sequence(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.residual_block(out_channels),
            self.residual_block(out_channels),
        )

    def compute_flattened_size(self, observation_space, cnn_channels):
        with torch.no_grad():
            sample_input = torch.zeros(self.ensemble_count, *observation_space).squeeze(
                -1
            )
            sample_output = self.conv_layers(sample_input).flatten(start_dim=1)
            return sample_output.shape[-1]

    def forward(self, observations: Tensor, actions: Tensor, feedback_type: str):
        """
        Forward pass through the network with feedback type conditioning.
        """
        # observations: (batch_size, segment_length, channels, height, width)
        # actions: (batch_size, segment_length, action_dim)
        batch_size, segment_length, channels, height, width = observations.shape
        _, _, action_dim = actions.shape

        # Get feedback type index and embedding
        feedback_idx = self.feedback_type_map[feedback_type]
        feedback_embedding = self.feedback_embedding(
            torch.tensor(feedback_idx, device=observations.device)
        )

        # Process observations through CNN
        obs_flat = observations.reshape(
            batch_size * segment_length, channels, height, width
        )
        x = self.conv_layers(obs_flat)
        x = self.flatten(x)
        x = F.relu(x)

        # Process actions
        actions_flat = actions.reshape(-1, action_dim)
        act = self.action_in(actions_flat)
        act = self.action_masksemble(act)
        act = F.relu(act)

        # Expand feedback embedding to match batch and sequence dimensions
        embedding_expanded = feedback_embedding.expand(batch_size * segment_length, -1)

        # Concatenate CNN features, action features, and feedback embedding
        combined = torch.cat((x, act, embedding_expanded), dim=1)

        # Final output
        output = self.fc(combined)

        # Reshape back
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def universal_loss(self, batch: Tensor):
        """
        A universal loss function that handles all feedback types.
        """
        feedback_type, data = batch
        feedback_type = feedback_types[0] if isinstance(feedback_types, list) else feedback_types
        feedback_idx = self.feedback_type_map[feedback_type]

        if feedback_type in [
            "comparative",
            "demonstrative",
            "corrective",
            "descriptive_preference",
        ]:
            # Handle pairwise comparison feedback
            pair_data, preferred_indices = data

            (obs1, actions1, mask1), (obs2, actions2, mask2) = pair_data

            # Compute network outputs for both trajectories
            outputs1 = self.forward(obs1, actions1, feedback_type)
            outputs2 = self.forward(obs2, actions2, feedback_type)

            # Sum over sequence dimension with masking
            rewards1 = (outputs1 * mask1).sum(dim=1).squeeze(-1)
            rewards2 = (outputs2 * mask2).sum(dim=1).squeeze(-1)

            # Apply learned normalization
            scale = torch.abs(self.loss_scale[feedback_idx]) + 1e-6  # Ensure positive
            bias = self.loss_bias[feedback_idx]

            normalized_rewards1 = rewards1 * scale + bias
            normalized_rewards2 = rewards2 * scale + bias

            # Stack rewards and compute log softmax
            rewards = torch.stack([normalized_rewards1, normalized_rewards2], dim=1)
            log_probs = F.log_softmax(rewards, dim=1)

            # Compute NLL loss
            loss = F.nll_loss(log_probs, preferred_indices)

        elif feedback_type in ["evaluative", "descriptive", "supervised"]:
            # Handle scalar feedback
            (observations, actions, masks), targets = data

            # Network output: (batch_size, segment_length, output_dim)
            outputs = self.forward(observations, actions, feedback_type)

            # Sum over the sequence dimension to get total rewards per segment
            total_rewards = (outputs * masks).sum(dim=1).squeeze(-1)

            # Apply learned normalization
            scale = torch.abs(self.loss_scale[feedback_idx]) + 1e-6  # Ensure positive
            bias = self.loss_bias[feedback_idx]

            normalized_rewards = total_rewards * scale + bias

            # Ensure targets have the correct shape
            targets = targets.float()
            if targets.dim() > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            # Compute MSE loss
            loss = F.mse_loss(normalized_rewards, targets)

        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

        return loss

    def training_step(self, batch: Tensor, batch_idx: int):
        """Compute universal loss for training."""
        loss = self.universal_loss(batch)

        feedback_type = batch[0]
        if isinstance(feedback_type, (list, tuple, torch.Tensor)):
            feedback_type = feedback_type[0]
        if isinstance(feedback_type, torch.Tensor):
            feedback_type = feedback_type.item()

        self.log(f"train_loss_{feedback_type}", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # Also log the learned normalization parameters
        if feedback_type in self.feedback_type_map:
            idx = self.feedback_type_map[feedback_type]
            self.log(f"norm_scale_{feedback_type}", self.loss_scale[idx], on_epoch=True)
            self.log(f"norm_bias_{feedback_type}", self.loss_bias[idx], on_epoch=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """Compute universal loss for validation."""
        loss = self.universal_loss(batch)

        feedback_type = batch[0]
        self.log(f"val_loss_{feedback_type}", loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
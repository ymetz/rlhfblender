from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from masksembles.torch import Masksembles1D, Masksembles2D
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.functional import log_softmax, mse_loss, nll_loss


class MultiHeadNetwork(LightningModule):
    """
    Network with shared backbone and multiple prediction heads for different feedback types.
    """

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        shared_layer_num: int = 5,  # Number of shared layers
        head_layer_num: int = 1,  # Number of layers in each head
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
    ):
        super().__init__()

        if feedback_types is None:
            feedback_types = ["evaluative", "comparative", "demonstrative", "descriptive"]

        self.feedback_types = feedback_types
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        self.masksemble_scale = masksemble_scale

        obs_space, action_space = input_spaces
        action_is_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Determine input dimension
        input_dim = np.prod(obs_space.shape) + (np.prod(action_space.shape) if not action_is_discrete else action_space.n)

        # Create shared backbone
        backbone_layers = []
        current_dim = input_dim

        for _ in range(shared_layer_num):
            backbone_layers.append(nn.Linear(current_dim, hidden_dim))
            backbone_layers.append(activation_function())

            if self.ensemble_count > 1:
                backbone_layers.append(
                    Masksembles1D(
                        channels=hidden_dim,
                        n=self.ensemble_count,
                        scale=self.masksemble_scale,
                    ).float()
                )

            current_dim = hidden_dim

        self.shared_backbone = nn.Sequential(*backbone_layers)

        # Create separate heads for each feedback type
        self.heads = nn.ModuleDict()

        for feedback_type in feedback_types:
            head_layers = []
            current_dim = hidden_dim

            # Add intermediate layers for the head
            for _ in range(head_layer_num - 1):
                head_layers.append(nn.Linear(current_dim, hidden_dim))
                head_layers.append(activation_function())

                if self.ensemble_count > 1:
                    head_layers.append(
                        Masksembles1D(
                            channels=hidden_dim,
                            n=self.ensemble_count,
                            scale=self.masksemble_scale,
                        ).float()
                    )

            # Add output layer
            head_layers.append(nn.Linear(current_dim, output_dim))

            if last_activation is not None:
                head_layers.append(last_activation())

                if self.ensemble_count > 1:
                    head_layers.append(
                        Masksembles1D(channels=output_dim, n=self.ensemble_count, scale=self.masksemble_scale).float()
                    )

            self.heads[feedback_type] = nn.Sequential(*head_layers)

        # Initialize weights
        self._init_weights()

        self.save_hyperparameters()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, observations: Tensor, actions: Tensor, feedback_type: str = None):
        """
        Forward pass through the network.
        If feedback_type is provided, only that head is used.
        Otherwise, all heads are used and a dictionary of outputs is returned.
        """
        # observations: (batch_size, segment_length, obs_dim)
        # actions: (batch_size, segment_length, action_dim)

        if len(observations.shape) > 3:
            observations = observations.flatten(start_dim=2)

        batch_size, segment_length, obs_dim = observations.shape
        _, _, action_dim = actions.shape

        # Flatten the batch and sequence dimensions
        obs_flat = observations.reshape(-1, obs_dim)
        actions_flat = actions.reshape(-1, action_dim)

        # Concatenate observations and actions
        x = torch.cat((obs_flat, actions_flat), dim=1)

        # Pass through shared backbone
        x = self.shared_backbone(x)

        # If specific feedback type is requested, only use that head
        if feedback_type is not None and feedback_type in self.heads:
            output = self.heads[feedback_type](x)
            output = output.reshape(batch_size, segment_length, -1)
            return output

        # Otherwise, use all heads and return a dictionary
        outputs = {}
        for fb_type, head in self.heads.items():
            head_output = head(x)
            outputs[fb_type] = head_output.reshape(batch_size, segment_length, -1)

        return outputs

    def _calculate_loss(self, in_data: Tensor, feedback_type: str):
        """Compute loss for training."""
        # Use the appropriate loss function for this feedback type
        if feedback_type in self.feedback_types:
            # For single-reward feedback types
            if feedback_type in ["evaluative", "descriptive", "supervised"]:
                data, targets = in_data

                observations, actions, masks = data[0], data[1], data[2]

                # Forward pass
                outputs = self.forward(observations, actions, feedback_type=feedback_type)

                # Sum over the sequence dimension to get total rewards per segment
                total_rewards = (outputs * masks).sum(dim=1).squeeze(-1)

                # Ensure targets have the correct shape
                targets = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)

                # Compute loss
                loss = nn.MSELoss()(total_rewards, targets)

            # For pairwise feedback types
            else:
                # Unpack data - assuming data is ((obs1, obs2), preferred_idx)
                trajectory_pair, preferred_idx = in_data

                # Extract observation pairs
                obs1, obs2 = trajectory_pair

                # Handle different formats - extract actions and masks if available
                obs1, actions1, mask1 = obs1[0], obs1[1], obs1[2]
                obs2, actions2, mask2 = obs2[0], obs2[1], obs2[2]

                # Forward pass for both trajectories
                outputs1 = self.forward(obs1, actions1, feedback_type=feedback_type)
                outputs2 = self.forward(obs2, actions2, feedback_type=feedback_type)

                # Sum over sequence dimension
                rewards1 = (outputs1 * mask1).sum(dim=1).squeeze(-1)
                rewards2 = (outputs2 * mask2).sum(dim=1).squeeze(-1)

                # Stack rewards and compute log softmax
                rewards = torch.stack([rewards1, rewards2], dim=1)  # Shape: (batch_size, 2)
                log_probs = F.log_softmax(rewards, dim=1)

                # Compute NLL loss
                loss = nll_loss(log_probs, preferred_idx)

            return loss
        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

    def training_step(self, batch, batch_idx):
        feedback_type, in_data = batch

        loss = self._calculate_loss(in_data, feedback_type[0])
        self.log(f"train_loss_{feedback_type}", loss, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        feedback_type, in_data = batch
        loss = self._calculate_loss(in_data, feedback_type[0])
        self.log(f"val_loss_{feedback_type}", loss, on_epoch=True, prog_bar=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

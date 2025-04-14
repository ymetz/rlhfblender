import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, depths=[16, 32, 32], **conv_kwargs):
        super(ImpalaCNN, self).__init__(observation_space, features_dim=256)
        self.depths = depths
        self.conv_kwargs = conv_kwargs
        self.layer_num = 0

        layers = []
        input_channels = observation_space.shape[0]

        for depth in depths:
            layers.append(self.conv_sequence(input_channels, depth))
            input_channels = depth

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.compute_flattened_size(observation_space), 256)

    def get_layer_num_str(self):
        num_str = str(self.layer_num)
        self.layer_num += 1
        return num_str

    def conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **self.conv_kwargs)

    def residual_block(self, in_channels):
        return nn.Sequential(
            nn.ReLU(),
            self.conv_layer(in_channels, in_channels),
            nn.ReLU(),
            self.conv_layer(in_channels, in_channels),
        )

    def conv_sequence(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.residual_block(out_channels),
            self.residual_block(out_channels),
        )

    def compute_flattened_size(self, observation_space):
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.conv_layers(sample_input)
            return int(np.prod(sample_output.size()))

    def forward(self, observations):
        x = observations / 255.0
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

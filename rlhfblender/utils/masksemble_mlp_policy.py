from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from masksembles.torch import Masksembles1D
from scipy.stats import mode
from stable_baselines3.common.distributions import (CategoricalDistribution,
                                                    DiagGaussianDistribution)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from torch import nn
from torch.distributions import Categorical, Normal

from .drop_connect import DropConnectLinear


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        mask_overlap: float = 2.0,
        num_masks: int = 4,
    ):
        super(CustomNetwork, self).__init__()
        device = get_device("auto")
        self.device = device

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.num_masks = num_masks
        self.mask_overlap = mask_overlap
        print(hidden_dim, last_layer_dim_pi, last_layer_dim_vf)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            Masksembles1D(hidden_dim, self.num_masks, self.mask_overlap, device=device),
            nn.Linear(hidden_dim, last_layer_dim_pi),
            nn.LeakyReLU(),
            Masksembles1D(
                last_layer_dim_pi, self.num_masks, self.mask_overlap, device=device
            ),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            Masksembles1D(hidden_dim, self.num_masks, self.mask_overlap, device=device),
            nn.Linear(hidden_dim, last_layer_dim_vf),
            nn.LeakyReLU(),
            Masksembles1D(
                last_layer_dim_vf, self.num_masks, self.mask_overlap, device=device
            ),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class MasksemblesMlpActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(MasksemblesMlpActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.is_masksemble = True

        self.net_arch = net_arch

    def _build_mlp_extractor(self) -> None:
        if self.net_arch is None:
            self.net_arch = {}
        self.mlp_extractor = CustomNetwork(self.features_dim, **self.net_arch)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        masksemble_mode: str = "MODE",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)
        # observation = th.tile(observation, (4*observation.shape[0], 1, 1, 1))
        observation = th.cat([observation] * self.mlp_extractor.num_masks)

        # For continuous observations, there is not MODE sampling, istead switch to AVERAGE sampling in the default case
        if (
            isinstance(self.action_dist, DiagGaussianDistribution)
            and masksemble_mode != "INITIAL"
        ):
            masksemble_mode = "AVERAGE"

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = self.get_distribution(observation).distribution.probs
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                means = self.get_distribution(observation).distribution.loc
                std = self.get_distribution(observation).distribution.scale
                probs = means
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = th.mean(probs, dim=0).cpu()
                actions = [Categorical(probs=probs).sample().numpy()]
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                avg_mean = th.mean(means, dim=0).cpu()
                avg_std = th.mean(std, dim=0).cpu()
                actions = [Normal(loc=avg_mean, scale=avg_std).sample().numpy()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[0].cpu().numpy()]
        else:  # masksemble_mode == "MODE"
            actions = actions.cpu().numpy()
            actions = mode(actions)
            actions = actions.mode

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
        return actions, state, probs


# =========== DropoutMlPolicy


class CustomDropoutNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        dropout_p: float = 0.2,
    ):
        super(CustomDropoutNetwork, self).__init__()
        device = get_device("auto")
        self.device = device

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.dropout_p = dropout_p

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim, last_layer_dim_pi),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim, last_layer_dim_vf),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class DropoutMlpActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(DropoutMlpActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.is_masksemble = True
        self.net_arch = net_arch

    def _build_mlp_extractor(self) -> None:
        if self.net_arch is None:
            self.net_arch = {}
        self.mlp_extractor = CustomDropoutNetwork(self.features_dim, **self.net_arch)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        masksemble_mode: str = "MODE",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(True)

        observation, vectorized_env = self.obs_to_tensor(observation)
        observation = th.tile(observation, (4 * observation.shape[0], 1, 1, 1))

        # For continuous observations, there is not MODE sampling, istead switch to AVERAGE sampling in the default case
        if (
            isinstance(self.action_dist, DiagGaussianDistribution)
            and masksemble_mode != "INITIAL"
        ):
            masksemble_mode = "AVERAGE"

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            if isinstance(self.get_distribution(observation).distribution, Categorical):
                probs = self.get_distribution(observation).distribution.probs
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                means = self.get_distribution(observation).distribution.loc
                std = self.get_distribution(observation).distribution.scale
                probs = means
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = th.mean(probs, dim=0).cpu()
                actions = [Categorical(probs=probs).sample().numpy()]
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                avg_mean = th.mean(means, dim=0).cpu()
                avg_std = th.mean(std, dim=0).cpu()
                actions = [Normal(loc=avg_mean, scale=avg_std).sample().numpy()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[0].cpu().numpy()]
        else:  # masksemble_mode == "MODE"
            actions = actions.cpu().numpy()
            actions = mode(actions)
            actions = actions.mode

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
        return actions, state, probs


# ==================== EnsemblePolicy


class CustomEnsembleNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        num_models: int = 4,
    ):
        super(CustomEnsembleNetwork, self).__init__()
        device = get_device("auto")
        self.device = device

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.num_models = num_models

        # Policy network
        self.policy_nets = [
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, last_layer_dim_pi),
                nn.LeakyReLU(),
            ).to(device)
            for _ in range(self.num_models)
        ]
        # Value network
        self.value_nets = [
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, last_layer_dim_vf),
                nn.LeakyReLU(),
            ).to(device)
            for _ in range(self.num_models)
        ]

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch = features.shape[0]
        x = th.split(features.unsqueeze(1), batch // self.num_models, dim=0)
        return (
            th.cat(
                [self.policy_nets[i](x[i]).squeeze(1) for i in range(self.num_models)],
                dim=0,
            ),
            th.cat(
                [self.value_nets[i](x[i]) for i in range(self.num_models)], dim=0
            ).squeeze(1),
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        batch = features.shape[0]
        x = th.split(features.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [self.policy_nets[i](x[i]).squeeze(1) for i in range(self.num_models)],
            dim=0,
        )

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch = features.shape[0]
        x = th.split(features.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [self.value_nets[i](x[i]) for i in range(self.num_models)], dim=0
        ).squeeze(1)


class EnsembleMlpActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(EnsembleMlpActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.is_masksemble = True
        self.net_arch = net_arch

    def _build_mlp_extractor(self) -> None:
        if self.net_arch is None:
            self.net_arch = {}
        self.mlp_extractor = CustomEnsembleNetwork(self.features_dim, **self.net_arch)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        masksemble_mode: str = "MODE",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)
        observation = th.tile(observation, (4 * observation.shape[0], 1, 1, 1))

        # For continuous observations, there is not MODE sampling, istead switch to AVERAGE sampling in the default case
        if (
            isinstance(self.get_distribution(observation).distribution, Normal)
            and masksemble_mode != "INITIAL"
        ):
            masksemble_mode = "AVERAGE"

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = self.get_distribution(observation).distribution.probs
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                means = self.get_distribution(observation).distribution.loc
                std = self.get_distribution(observation).distribution.scale
                probs = means
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = th.mean(probs, dim=0).cpu()
                actions = [Categorical(probs=probs).sample().numpy()]
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                avg_mean = th.mean(means, dim=0).cpu()
                avg_std = th.mean(std, dim=0).cpu()
                actions = [Normal(loc=avg_mean, scale=avg_std).sample().numpy()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[0].cpu().numpy()]
        else:  # masksemble_mode == "MODE"
            actions = actions.cpu().numpy()
            actions = mode(actions)
            actions = actions.mode

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
        return actions, state, probs


# =========== DropconnectMlpPolicy


class CustomDropConnectNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        activation_fn: Callable,
        hidden_dim: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        dropout_p: float = 0.2,
    ):
        super(CustomDropConnectNetwork, self).__init__()
        device = get_device("auto")
        self.device = device

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.dropout_p = dropout_p

        # Policy network
        self.policy_net = nn.Sequential(
            DropConnectLinear(feature_dim, hidden_dim, weight_dropout=dropout_p),
            activation_fn(),
            DropConnectLinear(hidden_dim, last_layer_dim_pi, weight_dropout=dropout_p),
            activation_fn(),
        )
        # Value network
        self.value_net = nn.Sequential(
            DropConnectLinear(feature_dim, hidden_dim, weight_dropout=dropout_p),
            activation_fn(),
            DropConnectLinear(hidden_dim, last_layer_dim_vf, weight_dropout=dropout_p),
            activation_fn(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class DropConnectMlpActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(DropConnectMlpActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.is_masksemble = True
        self.net_arch = net_arch

    def _build_mlp_extractor(self) -> None:
        if self.net_arch is None:
            self.net_arch = {}
        self.mlp_extractor = CustomDropConnectNetwork(
            self.features_dim, self.activation_fn, **self.net_arch
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        masksemble_mode: str = "AVERAGE",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(True)

        observation, vectorized_env = self.obs_to_tensor(observation)
        observation = th.tile(observation, (4 * observation.shape[0], 1, 1, 1))

        # For continuous observations, there is not MODE sampling, istead switch to AVERAGE sampling in the default case
        if (
            isinstance(self.get_distribution(observation).distribution, Normal)
            and masksemble_mode != "INITIAL"
        ):
            masksemble_mode = "AVERAGE"

        with th.no_grad():
            actions = th.cat(
                [self._predict(o, deterministic=deterministic) for o in observation]
            )
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = th.cat(
                    [self.get_distribution(o).distribution.probs for o in observation]
                )
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                means = th.cat(
                    [self.get_distribution(o).distribution.loc for o in observation]
                )
                std = th.cat(
                    [self.get_distribution(o).distribution.scale for o in observation]
                )
                probs = means
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            if isinstance(self.action_dist, CategoricalDistribution):
                probs = th.mean(probs, dim=0).cpu()
                actions = [Categorical(probs=probs).sample().numpy()]
            elif isinstance(self.action_dist, DiagGaussianDistribution):
                avg_mean = th.mean(means, dim=0).cpu()
                avg_std = th.mean(std, dim=0).cpu()
                actions = [Normal(loc=avg_mean, scale=avg_std).sample().numpy()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[0].cpu().numpy()]
        else:  # masksemble_mode == "MODE"
            actions = actions.cpu().numpy()
            actions = mode(actions)
            actions = actions.mode

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
        return actions, state, probs

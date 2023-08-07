import collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from masksembles.torch import Masksembles1D, Masksembles2D
from scipy.stats import mode
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from torch import nn
from torch.distributions.categorical import Categorical

from .bam_layer import BAM
from .drop_connect import DropConnectConv2d, DropConnectLinear


class BamCNNNetwork(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        shared_bam_policy_path: str = "",
        shared_bam_layer=None,
        shared_feature_extractor=None,
    ):
        super(BamCNNNetwork, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        if shared_feature_extractor:
            print("[Forward PPO] uses shared feature extractor")
            self.cnn = shared_feature_extractor
        elif shared_bam_policy_path != "":
            print("uses shared bam layer")

            print("[INFO] ----- Use a loaded and frozen shared BAM layer")
            shared_bam_layer = th.load(
                shared_bam_policy_path, map_location=th.device("cpu")
            )
            print(
                "[INFO] ---- Loaded Norm:",
                th.norm(
                    th.nn.utils.parameters_to_vector(
                        [p for p in shared_bam_layer.parameters() if p.requires_grad]
                    )
                ).item(),
            )
            for p in shared_bam_layer.parameters():
                p.requires_grad = False

            self.cnn = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "Conv_2",
                            nn.Conv2d(32, 48, kernel_size=8, stride=4, padding=0),
                        ),
                        ("Relu_2", nn.ReLU()),
                        (
                            "Conv_3",
                            nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=0),
                        ),
                        ("Reul_3", nn.ReLU()),
                        # BAM(64),
                        ("Flatten", nn.Flatten()),
                    ]
                )
            )
            self.cnn = nn.Sequential(*(list(shared_bam_layer) + list(self.cnn)))
        else:
            self.cnn = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "Conv_1",
                            nn.Conv2d(
                                n_input_channels, 32, kernel_size=3, stride=1, padding=0
                            ),
                        ),
                        ("Relu_1", nn.ReLU()),
                        ("BAM_1", BAM(32)),
                        (
                            "Conv_2",
                            nn.Conv2d(32, 48, kernel_size=8, stride=4, padding=0),
                        ),
                        ("Relu_2", nn.ReLU()),
                        (
                            "Conv_3",
                            nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=0),
                        ),
                        ("Reul_3", nn.ReLU()),
                        ("Flatten", nn.Flatten()),
                    ]
                )
            )

        # Compute shape by doing one forward pass
        print(
            "[Forward Policy] Number of paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters()),
        )
        for layer in self.cnn:
            if not (isinstance(layer, nn.ReLU) or isinstance(layer, nn.Flatten)):
                print("{} parameters".format(layer), next(layer.parameters()).device)
        print(
            "Input Tensor device",
            th.as_tensor([observation_space.sample() for _ in range(2)]).float().device,
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor([observation_space.sample() for _ in range(2)]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print(
            "================° [Forward Policy: BamCNN] Total paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters())
            + sum(p.numel() for p in self.linear.parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def interfered_forward(self, input: th.Tensor) -> th.Tensor():
        return self.linear(self.cnn[2:](input))


class BamActorCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = BamCNNNetwork,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(BamActorCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


# ============= Larger NatureCNN ===========
class LargeNatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(LargeNatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Conv_1",
                        nn.Conv2d(
                            n_input_channels, 32, kernel_size=3, stride=1, padding=0
                        ),
                    ),
                    ("Relu_1", nn.ReLU()),
                    ("Conv_2", nn.Conv2d(32, 48, kernel_size=8, stride=4, padding=0)),
                    ("Relu_2", nn.ReLU()),
                    ("Conv_3", nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=0)),
                    ("Reul_3", nn.ReLU()),
                    # BAM(64),
                    ("Flatten", nn.Flatten()),
                ]
            )
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        print(
            "================° [Forward Policy: LargeNatureCNN] Total paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters())
            + sum(p.numel() for p in self.linear.parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class LargeNatureCNNCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = LargeNatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(LargeNatureCNNCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


# ============= MasksemblesCNN ===========
class MasksemblesCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Extended with Masksembles
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        num_masks: int = 4,
    ):
        super(MasksemblesCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        self.num_masks = num_masks
        device = get_device("auto")
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Conv_1",
                        nn.Conv2d(
                            n_input_channels, 32, kernel_size=8, stride=4, padding=0
                        ),
                    ),
                    ("Relu_1", nn.LeakyReLU()),
                    ("Conv_2", nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
                    ("Relu_2", nn.LeakyReLU()),
                    ("Masksembles_1", Masksembles2D(64, num_masks, 3.0, device=device)),
                    ("Conv_3", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
                    ("Relu_3", nn.LeakyReLU()),
                    ("Flatten", nn.Flatten()),
                    (
                        "Masksembles_2",
                        Masksembles1D(3136, num_masks, 3.0, device=device),
                    ),
                ]
            )
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor([observation_space.sample() for _ in range(8)]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())
        print(
            "================° [Forward Policy] Total paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters())
            + sum(p.numel() for p in self.linear.parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MasksemblesCNNCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = MasksemblesCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(MasksemblesCNNCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.is_masksemble = True

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

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            probs = self.get_distribution(observation).distribution.probs
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            action_probs = th.mean(probs, dim=0).cpu()
            if deterministic:
                actions = [Categorical(probs=action_probs).sample().numpy()]
            else:
                actions = [Categorical(probs=action_probs).mode()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[2]]
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


# === DropoutCNN =========
class DropoutCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Extended with Masksembles
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        dropout_p: float = 0.2,
    ):
        super(DropoutCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        get_device("auto")
        self.dropout_p = dropout_p
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Conv_1",
                        nn.Conv2d(
                            n_input_channels, 32, kernel_size=8, stride=4, padding=0
                        ),
                    ),
                    ("Relu_1", nn.LeakyReLU()),
                    ("Conv_2", nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
                    ("Relu_2", nn.LeakyReLU()),
                    ("Dropout_1", nn.Dropout2d(p=dropout_p)),
                    ("Conv_3", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
                    ("Relu_3", nn.LeakyReLU()),
                    ("Flatten", nn.Flatten()),
                    ("Dropout_2", nn.Dropout(p=dropout_p)),
                ]
            )
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor([observation_space.sample() for _ in range(8)]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())
        print(
            "================° [Forward Policy] Total paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters())
            + sum(p.numel() for p in self.linear.parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class DropoutCNNCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = DropoutCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DropoutCNNCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.is_masksemble = True

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

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            probs = self.get_distribution(observation).distribution.probs
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            action_probs = th.mean(probs, dim=0).cpu()
            if deterministic:
                actions = [Categorical(probs=action_probs).sample().numpy()]
            else:
                actions = [Categorical(probs=action_probs).mode()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[2]]
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


# =================== EnsembleCNN Policy

# === DropoutCNN =========
class EnsembleCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Extended with Masksembles
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        num_models: int = 4,
    ):
        super(EnsembleCNN, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )

        self.num_models = num_models
        self.features_dim = features_dim
        self._observation_space = observation_space
        # Save dim, used to create the distributions
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

        device = get_device("auto")
        self.device = device
        n_input_channels = observation_space.shape[0]
        self.cnns = [
            nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "Conv_1",
                            nn.Conv2d(
                                n_input_channels, 32, kernel_size=8, stride=4, padding=0
                            ),
                        ),
                        ("Relu_1", nn.LeakyReLU()),
                        (
                            "Conv_2",
                            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        ),
                        ("Relu_2", nn.LeakyReLU()),
                        ("Dropout_1", nn.Dropout2d(p=0.2)),
                        (
                            "Conv_3",
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                        ),
                        ("Relu_3", nn.LeakyReLU()),
                        ("Flatten", nn.Flatten()),
                        ("Dropout_2", nn.Dropout(p=0.2)),
                    ]
                )
            ).to(device)
            for _ in range(self.num_models)
        ]

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnns[0](
                th.as_tensor([observation_space.sample() for _ in range(8)])
                .to(device)
                .float()
            ).shape[1]

        self.linears = [
            nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU()).to(device)
            for _ in range(self.num_models)
        ]

        print(
            "================° [Forward Policy] Total paramters in CNN:",
            sum(p.numel() for p in self.cnns[0].parameters())
            + sum(p.numel() for p in self.linears[0].parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        x = th.split(observations.unsqueeze(1), batch // self.num_models, dim=0)
        common_features = th.cat(
            [
                self.linears[i](self.cnns[i](x[i].squeeze(1)))
                for i in range(self.num_models)
            ],
            dim=0,
        )
        return common_features, common_features

    def forward_actor(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        x = th.split(observations.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [
                self.linears[i](self.cnns[i](x[i].squeeze(1)))
                for i in range(self.num_models)
            ],
            dim=0,
        )

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch = features.shape[0]
        x = th.split(observations.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [
                self.linears[i](self.cnns[i](x[i].squeeze(1)))
                for i in range(self.num_models)
            ],
            dim=0,
        )


class EnsembleValueNet(nn.Module):
    def __init__(self, in_feature_dim: int = 64, num_models: int = 4):
        super(EnsembleValueNet, self).__init__()
        self.num_models = num_models

        device = get_device("auto")
        self.value_outs = [
            nn.Linear(in_feature_dim, 1).to(device) for _ in range(self.num_models)
        ]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        x = th.split(observations.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [self.value_outs[i](x[i].squeeze(1)) for i in range(self.num_models)], dim=0
        )


class EnsembleActionNet(nn.Module):
    def __init__(self, in_feature_dim: int = 64, num_models: int = 4):
        super(EnsembleActionNet, self).__init__()
        self.num_models = num_models

        device = get_device("auto")
        self.action_outs = [
            nn.Linear(in_feature_dim, 1).to(device) for _ in range(self.num_models)
        ]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        x = th.split(observations.unsqueeze(1), batch // self.num_models, dim=0)
        return th.cat(
            [self.action_outs[i](x[i].squeeze(1)) for i in range(self.num_models)],
            dim=0,
        )


class IdentityExtractor(nn.Module):
    """
    A bit of a hack to go around SB3s design. We do not use a specific feature extractor, so just pass through the values
    """

    def __init__(self, observation_space, features_dim: int = 512, **kwargs):
        super(IdentityExtractor, self).__init__()

        self.features_dim = features_dim

    def forward(self, x):
        return x


class EnsembleCNNCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = IdentityExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        fully_independent_networks: bool = False,
        num_models: int = 4,
    ):
        super(EnsembleCNNCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.is_masksemble = True
        self.net_arch = net_arch
        self.num_models = num_models

        print("self device", self.device)

        # ======= Overwrite value and action net to get a fully independent ensemble, currently not fully implemented
        if fully_independent_networks:
            self.value_net = EnsembleValueNet(self.features_dim, self.num_models)
            self.action_net = EnsembleActionNet(self.features_dim, self.num_models)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = EnsembleCNN(
            self.observation_space, self.features_dim, **self.features_extractor_kwargs
        )

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

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            probs = self.get_distribution(observation).distribution.probs
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            action_probs = th.mean(probs, dim=0).cpu()
            if deterministic:
                actions = [Categorical(probs=action_probs).sample().numpy()]
            else:
                actions = [Categorical(probs=action_probs).mode()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[2]]
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


# ============= DropConnect Policy ===============
class DropConnectCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Extended with Masksembles
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        dropout_p: float = 0.2,
    ):
        super(DropConnectCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        get_device("auto")
        self.dropout_p = dropout_p
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Conv_1",
                        nn.Conv2d(
                            n_input_channels, 32, kernel_size=8, stride=4, padding=0
                        ),
                    ),
                    ("Relu_1", nn.LeakyReLU()),
                    ("Conv_2", nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
                    ("Relu_2", nn.LeakyReLU()),
                    (
                        "Conv_3",
                        DropConnectConv2d(
                            64,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            weight_dropout=dropout_p,
                        ),
                    ),
                    ("Relu_3", nn.LeakyReLU()),
                    ("Flatten", nn.Flatten()),
                ]
            )
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor([observation_space.sample() for _ in range(8)]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            DropConnectLinear(n_flatten, features_dim, weight_dropout=dropout_p),
            nn.LeakyReLU(),
        )
        print(
            "================° [Forward Policy] Total paramters in CNN:",
            sum(p.numel() for p in self.cnn.parameters())
            + sum(p.numel() for p in self.linear.parameters()),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class DropConnectCNNCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = DropConnectCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DropConnectCNNCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.is_masksemble = True

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

        with th.no_grad():
            actions = th.cat(
                [
                    self._predict(o.unsqueeze(0), deterministic=deterministic)
                    for o in observation
                ]
            )
            # print("get distribution in masksemble predict", self.get_distribution(observation).distribution.probs)
            probs = th.cat(
                [
                    self.get_distribution(o.unsqueeze(0)).distribution.probs
                    for o in observation
                ]
            )
            # _, probs, entropy = self.evaluate_actions(
            #     # Currently only works with discrete actions and image observations
            #     observation,
            #     th.from_numpy(np.tile(np.arange(self.action_space.n), (observation.shape[0], 1))).to(self.device)
            # )

        if masksemble_mode == "AVERAGE":
            action_probs = th.mean(probs, dim=0).cpu()
            if deterministic:
                actions = [Categorical(probs=action_probs).sample().numpy()]
            else:
                actions = [Categorical(probs=action_probs).mode()]
        elif masksemble_mode == "INITIAL":
            actions = [actions[2]]
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

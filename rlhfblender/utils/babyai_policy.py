from typing import Dict, List, Optional, Type

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)

    def forward(self, inputs):
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class FiLMBlock(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class BabyAIFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor adapted from BabyAI's ACModel for use with StableBaselines3.
    Combines image processing with instruction processing using FiLM.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_dim: int = 128,
        instr_dim: int = 128,
        use_instr: bool = True,
        lang_model: str = "gru",
        use_memory: bool = False,
        arch: str = "bow_endpool_res",
    ):
        super().__init__(observation_space, features_dim=image_dim)

        self.image_dim = image_dim
        self.instr_dim = instr_dim
        self.use_instr = use_instr
        self.lang_model = lang_model
        self.use_memory = use_memory
        self.arch = arch

        endpool = "endpool" in arch
        use_bow = "bow" in arch
        pixel = "pixel" in arch
        self.res = "res" in arch

        # Image processing
        image_space = observation_space.spaces["image"]
        self.image_conv = nn.Sequential(
            *[
                *([ImageBOWEmbedding(image_space.high.max(), 128)] if use_bow else []),
                *([nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(8, 8), stride=8, padding=0)] if pixel else []),
                nn.Conv2d(
                    in_channels=128 if use_bow or pixel else 3,
                    out_channels=128,
                    kernel_size=(3, 3) if endpool else (2, 2),
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            ]
        )
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Instruction processing
        if self.use_instr:
            instr_space = observation_space.spaces["instr"]
            self.word_embedding = nn.Embedding(instr_space.high.max() + 1, self.instr_dim)
            if self.lang_model in ["gru", "bigru", "attgru"]:
                gru_dim = self.instr_dim
                if self.lang_model in ["bigru", "attgru"]:
                    gru_dim //= 2
                self.instr_rnn = nn.GRU(
                    self.instr_dim, gru_dim, batch_first=True, bidirectional=(self.lang_model in ["bigru", "attgru"])
                )
                self.final_instr_dim = self.instr_dim

            if self.lang_model == "attgru":
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            # FiLM layers
            num_module = 2
            self.controllers = nn.ModuleList(
                [
                    FiLMBlock(
                        in_features=self.final_instr_dim,
                        out_features=128 if ni < num_module - 1 else self.image_dim,
                        in_channels=128,
                        imm_channels=128,
                    )
                    for ni in range(num_module)
                ]
            )

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == "gru":
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths - 1, :]
            return hidden

        elif self.lang_model in ["bigru", "attgru"]:
            embeddings = self.word_embedding(instr)
            if lengths.shape[0] > 1:
                embeddings = nn.utils.rnn.pack_padded_sequence(
                    embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
                )

            outputs, final_states = self.instr_rnn(embeddings)

            if lengths.shape[0] > 1:
                outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            if self.lang_model == "attgru":
                return outputs
            else:
                return final_states.transpose(0, 1).reshape(final_states.shape[1], -1)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        x = observations["image"]
        x = x.transpose(1, 3).transpose(2, 3)  # Convert to BCHW format

        # Process image
        x = self.image_conv(x)

        # Process instructions and apply FiLM if needed
        if self.use_instr:
            instr_embedding = self._get_instr_embedding(observations["instr"])
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out = out + x
                x = out

        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        return x


class BabyAIRecurrentPolicy(RecurrentActorCriticPolicy):
    """
    A recurrent policy for BabyAI environments that uses the custom features extractor
    and adapts it to StableBaselines3's architecture.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        image_dim: int = 128,
        instr_dim: int = 128,
        use_instr: bool = True,
        lang_model: str = "gru",
        arch: str = "bow_endpool_res",
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Custom initialization for BabyAI features extractor
        self.features_extractor_class = BabyAIFeaturesExtractor
        self.features_extractor_kwargs = dict(
            image_dim=image_dim, instr_dim=instr_dim, use_instr=use_instr, lang_model=lang_model, arch=arch
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
            **kwargs
        )

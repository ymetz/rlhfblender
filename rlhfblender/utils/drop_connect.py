import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd, _pair


def _weight_drop(module, weights, dropout):
    """
    Helper for 'DropConnect'.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + "_raw", Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + "_raw")
            w = torch.nn.functional.dropout(raw_w, p=dropout)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, "forward", forward)


class DropConnectLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ["weight"]
        _weight_drop(self, weights, weight_dropout)


class DropConnectConv2d(torch.nn.Conv2d):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ["weight"]
        _weight_drop(self, weights, weight_dropout)

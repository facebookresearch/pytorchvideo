# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from fvcore.nn.squeeze_excitation import SqueezeExcitation as SqueezeExcitationFVCore
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)

from .conv_helper import _Reshape, _SkipConnectMul


class SqueezeExcitation(EfficientBlockBase):
    """
    Efficient Squeeze-Excitation (SE). The Squeeze-Excitation block is described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    This implementation has the same instantiation interface as SE implementation in
    fvcore, and in original mode for training it is just a wrapped version of SE in
    fvcore. Since conv3d in original SE implementation of fvcore is not well supported
    by QNNPACK, here convert() method is implemented which converts class instance into
    a equivalent efficient deployable form.

    convert_flag variable is to record whether the SqueezeExcitation instance
    has been converted; SqueezeExcitation is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.
    """

    def __init__(
        self,
        num_channels: int,
        num_channels_reduced: Optional[int] = None,
        reduction_ratio: float = 2.0,
        is_3d: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()
        # Implement SE from FVCore here for training.
        self.se = SqueezeExcitationFVCore(
            num_channels,
            num_channels_reduced=num_channels_reduced,
            reduction_ratio=reduction_ratio,
            is_3d=is_3d,
            activation=activation,
        )
        self.is_3d = is_3d
        self.convert_flag = False

    def convert(self, input_blob_size, **kwargs):
        """
        Converts into efficient version of squeeze-excite (SE) for CPU.
        It changes conv in original SE into linear layer (better supported by CPU).
        """
        if self.is_3d:
            avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            avg_pool = nn.AdaptiveAvgPool2d(1)
        """
        Reshape tensor size to (B, C) for linear layer.
        """
        reshape0 = _Reshape((input_blob_size[0], input_blob_size[1]))
        fc0 = nn.Linear(
            self.se.block[0].in_channels,
            self.se.block[0].out_channels,
            bias=(not (self.se.block[0].bias is None)),
        )
        state_dict_fc0 = deepcopy(self.se.block[0].state_dict())
        state_dict_fc0["weight"] = state_dict_fc0["weight"].squeeze()
        fc0.load_state_dict(state_dict_fc0)
        activation = deepcopy(self.se.block[1])
        fc1 = nn.Linear(
            self.se.block[2].in_channels,
            self.se.block[2].out_channels,
            bias=(not (self.se.block[2].bias is None)),
        )
        state_dict_fc1 = deepcopy(self.se.block[2].state_dict())
        state_dict_fc1["weight"] = state_dict_fc1["weight"].squeeze()
        fc1.load_state_dict(state_dict_fc1)
        sigmoid = deepcopy(self.se.block[3])
        """
        Output of linear layer has output shape of (B, C). Need to reshape to proper
        shape before multiplying with input tensor.
        """
        reshape_size_after_sigmoid = (input_blob_size[0], input_blob_size[1], 1, 1) + (
            (1,) if self.is_3d else ()
        )
        reshape1 = _Reshape(reshape_size_after_sigmoid)
        se_layers = nn.Sequential(
            avg_pool, reshape0, fc0, activation, fc1, sigmoid, reshape1
        )
        # Add final elementwise multiplication and replace self.se
        self.se = _SkipConnectMul(se_layers)
        self.convert_flag = True

    def forward(self, x) -> torch.Tensor:
        out = self.se(x)
        return out

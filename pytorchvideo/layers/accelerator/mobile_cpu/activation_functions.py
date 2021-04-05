# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This file contains supported activation functions in efficient block and helper code.
All supported activation functions are child class of EfficientBlockBase, and included
in supported_act_functions.
"""
import torch
import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)
from pytorchvideo.layers.swish import Swish as SwishCustomOp


class _NaiveSwish(nn.Module):
    """
    Helper class to implement naive swish for deploy. It is not intended to be used to
    build network.
    """

    def __init__(self):
        super().__init__()
        self.mul_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.mul_func.mul(x, torch.sigmoid(x))


class Swish(EfficientBlockBase):
    """
    Swish activation function for efficient block. When in original form for training,
    using custom op version of swish for better training memory efficiency. When in
    deployable form, use naive swish as custom op is not supported to run on Pytorch
    Mobile. For better latency on mobile CPU, use HardSwish instead.
    """

    def __init__(self):
        super().__init__()
        self.act = SwishCustomOp()

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        self.act = _NaiveSwish()


class HardSwish(EfficientBlockBase):
    """
    Hardswish activation function. It is natively supported by Pytorch Mobile, and has
    better latency than Swish in int8 mode.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


class ReLU(EfficientBlockBase):
    """
    ReLU activation function for EfficientBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


class Identity(EfficientBlockBase):
    """
    Identity operation for EfficientBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


supported_act_functions = {
    "relu": ReLU,
    "swish": Swish,
    "hswish": HardSwish,
    "identity": Identity,
}

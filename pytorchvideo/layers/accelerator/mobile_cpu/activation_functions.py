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
    Helper class to implement the naive Swish activation function for deployment.
    It is not intended to be used to build networks.

    Swish(x) = x * sigmoid(x)

    Args:
        None

    Returns:
        torch.Tensor: The output tensor after applying the naive Swish activation.
    """

    def __init__(self):
        super().__init__()
        self.mul_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Forward pass through the naive Swish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the naive Swish activation.
        """
        return self.mul_func.mul(x, torch.sigmoid(x))


class Swish(EfficientBlockBase):
    """
    Swish activation function for efficient block.

    When in its original form for training, it uses a custom op version of Swish for
    better training memory efficiency. When in a deployable form, it uses a naive Swish
    as the custom op is not supported for running on PyTorch Mobile. For better latency
    on mobile CPU, consider using HardSwish instead.

    Args:
        None

    Returns:
        torch.Tensor: The output tensor after applying the Swish activation.
    """

    def __init__(self):
        super().__init__()

        # Initialize the activation function based on whether it's for training or deployment
        self.act = SwishCustomOp()  # Use SwishCustomOp if defined, otherwise use _NaiveSwish

    def forward(self, x):
        """
        Forward pass through the Swish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Swish activation.
        """
        return self.act(x)

    def convert(self, *args, **kwargs):
        """
        Convert the activation function to use naive Swish for deployment.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            None
        """
        self.act = _NaiveSwish()


class HardSwish(EfficientBlockBase):
    """
    Hardswish activation function. It is natively supported by PyTorch Mobile and has
    better latency than Swish in int8 mode.

    Args:
        None

    Returns:
        torch.Tensor: The output tensor after applying the HardSwish activation.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Hardswish()

    def forward(self, x):
        """
        Forward pass through the HardSwish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the HardSwish activation.
        """
        return self.act(x)

    def convert(self, *args, **kwargs):
        """
        Placeholder method for converting the activation function. No conversion is
        performed for HardSwish.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            None
        """
        pass


class ReLU(EfficientBlockBase):
    """
    ReLU activation function for EfficientBlockBase.

    Args:
        None

    Returns:
        torch.Tensor: The output tensor after applying the ReLU activation.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the ReLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the ReLU activation.
        """
        return self.act(x)

    def convert(self, *args, **kwargs):
        """
        Placeholder method for converting the activation function. No conversion is
        performed for ReLU.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            None
        """
        pass


class Identity(EfficientBlockBase):
    """
    Identity operation for EfficientBlockBase. It simply returns the input tensor
    unchanged.

    Args:
        None

    Returns:
        torch.Tensor: The input tensor itself.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the Identity operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor itself.
        """
        return self.act(x)

    def convert(self, *args, **kwargs):
        """
        Placeholder method for converting the identity operation. No conversion is
        performed for Identity.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            None
        """
        pass



supported_act_functions = {
    "relu": ReLU,
    "swish": Swish,
    "hswish": HardSwish,
    "identity": Identity,
}

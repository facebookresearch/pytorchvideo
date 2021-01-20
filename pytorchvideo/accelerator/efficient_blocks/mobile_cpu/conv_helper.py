# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This file contains helper classes for building conv3d efficient blocks.
The helper classes are intended to be instantiated inside efficient block,
not to be used by user to build network.
"""

from typing import Tuple

import torch
import torch.nn as nn


class _Reshape(nn.Module):
    """
    Helper class to implement data reshape as a module.
    Args:
        reshape_size (tuple): size of data after reshape.
    """

    def __init__(
        self,
        reshape_size: Tuple,
    ):
        super().__init__()
        self.reshape_size = reshape_size

    def forward(self, x):
        return torch.reshape(x, self.reshape_size)

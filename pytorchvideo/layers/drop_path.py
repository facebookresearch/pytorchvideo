# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Stochastic Depth per sample.

    Args:
        x (tensor): Input tensor.
        drop_prob (float): Probability to apply drop path.
        training (bool): If True, apply drop path to input. Otherwise (tesing), return input.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Args:
            drop_prob (float): Probability to apply drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        return drop_path(x, self.drop_prob, self.training)

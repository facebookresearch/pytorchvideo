# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Apply stochastic depth regularization to the input tensor.

    Stochastic Depth is a regularization technique used in deep neural networks.
    During training, it randomly drops (sets to zero) a fraction of the input tensor
    elements to prevent overfitting. During inference, no elements are dropped.

    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float): Probability to apply drop path (0.0 means no drop).
        training (bool): If True, apply drop path during training; otherwise, return the input.

    Returns:
        torch.Tensor: Output tensor after applying drop path.
    """
    if drop_prob == 0.0 or not training:
        # If drop probability is 0 or not in training mode, return the input as is.
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Adjust shape for various tensor dimensions.
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # Binarize the mask.

    # Scale the input tensor and apply the mask to drop elements.
    output = x.div(keep_prob) * mask

    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Drop path is a regularization technique used in deep neural networks to
    randomly drop (set to zero) a fraction of input tensor elements during training
    to prevent overfitting.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Initialize the DropPath module.

        Args:
            drop_prob (float): Probability to apply drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply drop path regularization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying drop path.
        """
        # Call the drop_path function to apply drop path to the input tensor.
        return drop_path(x, self.drop_prob, self.training)

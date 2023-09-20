# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, List

import torch
import torch.nn as nn


"""
Fusion layers are nn.Modules that take a list of Tensors (e.g. from a multi-stream
architecture), and return a single fused Tensor. This file has several
different types of fusion layers and a factory function "make_fusion_layer" to
construct them.
"""


def make_fusion_layer(method: str, feature_dims: List[int]):
    """
    Drop paths (Stochastic Depth) per sample.

    Drop path is a regularization technique used in deep neural networks to
    randomly drop (set to zero) a fraction of input tensor elements during training
    to prevent overfitting.

    Args:
        method (str): the fusion method to be constructed. Options:
            - 'concat'
            - 'temporal_concat'
            - 'max'
            - 'sum'
            - 'prod'

        feature_dims (List[int]): the first argument of all fusion layers. It holds a list
            of required feature_dims for each tensor input (where the tensor inputs are of
            shape (batch_size, seq_len, feature_dim)). The list order must corresponds to
            the tensor order passed to forward(...).

    Example:
        >>> drop_path_layer = DropPath(drop_prob=0.2)
        >>> input_tensor = torch.rand(1, 64, 128, 128)  # Example input tensor
        >>> output_tensor = drop_path_layer(input_tensor)
    """
    if method == "concat":
        return ConcatFusion(feature_dims)
    elif method == "temporal_concat":
        return TemporalConcatFusion(feature_dims)
    elif method == "max":
        return ReduceFusion(feature_dims, lambda x: torch.max(x, dim=0).values)
    elif method == "sum":
        return ReduceFusion(feature_dims, lambda x: torch.sum(x, dim=0))
    elif method == "prod":
        return ReduceFusion(feature_dims, lambda x: torch.prod(x, dim=0))
    else:
        raise NotImplementedError(f"Fusion {method} not available.")


class ConcatFusion(nn.Module):
    """
    Concatenates multiple input tensors along their last dimension to create a fused tensor.
    The size of the last dimension in the resulting tensor is the sum of the last dimensions
    of all input tensors.

    Args:
        feature_dims (List[int]): A list of feature dimensions for each input tensor.

    Attributes:
        output_dim (int): The size of the last dimension in the fused tensor.

    Example:
        If feature_dims is [64, 128, 256], and three tensors of shape (batch_size, seq_len, 64),
        (batch_size, seq_len, 128), and (batch_size, seq_len, 256) are concatenated, the output
        tensor will have a shape of (batch_size, seq_len, 448) because 64 + 128 + 256 = 448.
    """

    def __init__(self, feature_dims: List[int]):
        """
        Initialize the ConcatFusion module.

        Args:
            feature_dims (List[int]): A list of feature dimensions for each input tensor.
        """
        super().__init__()
        _verify_feature_dim(feature_dims)
        self._output_dim = sum(feature_dims)

    @property
    def output_dim(self):
        """
        Get the size of the last dimension in the fused tensor.

        Returns:
            int: The size of the last dimension in the fused tensor.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate a list of input tensors along their last dimension.

        Args:
            input_list (List[torch.Tensor]): A list of tensors to be concatenated.

        Returns:
            torch.Tensor: A tensor resulting from the concatenation of input tensors.
                The size of the last dimension is the sum of the feature dimensions
                of all input tensors.
        """
        return torch.cat(input_list, dim=-1)


class TemporalConcatFusion(nn.Module):
    """
    Concatenates input tensors along their temporal dimension (assumed to be dim=1).

    This module takes a list of input tensors, each with shape (batch_size, seq_len, feature_dim),
    and concatenates them along the temporal dimension (dim=1).

    Args:
        feature_dims (List[int]): List of feature dimensions of the input tensors.

    Note:
        - All input tensors must have the same feature dimension.
        - The output tensor will have shape (batch_size, sum(seq_len), feature_dim),
          where sum(seq_len) is the sum of seq_len for all input tensors.
    """

    def __init__(self, feature_dims: List[int]):
        """
        Initialize the TemporalConcatFusion module.

        Args:
            feature_dims (List[int]): List of feature dimensions of the input tensors.
        """
        super().__init__()
        _verify_feature_dim(feature_dims)

        # All input dimensions must be the same
        self._output_dim = max(feature_dims)
        assert self._output_dim == min(feature_dims)

    @property
    def output_dim(self):
        """
        Get the last dimension size of the output tensor produced by the forward(..) method.

        Returns:
            int: Last dimension size of the forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform forward pass through the TemporalConcatFusion module.

        Args:
            input_list (List[torch.Tensor]): A list of tensors of shape
                (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sum(seq_len), feature_dim),
                where sum(seq_len) is the sum of all input tensors' seq_len.
        """
        return torch.cat(input_list, dim=1)


class ReduceFusion(nn.Module):
    """
    A generic fusion method that applies a specified reduction function to a list of input tensors
    to produce a single output tensor. This class can be used to implement fusion methods like "sum",
    "max", and "prod".

    Args:
        feature_dims (List[int]): List of feature dimensions for the input tensors.
        reduce_fn (Callable[[torch.Tensor], torch.Tensor]): A callable reduction function that takes
            the list of input tensors and returns a single tensor.

    Attributes:
        output_dim (int): The dimension of the output tensor after fusion, which is the maximum
            of the input feature dimensions.

    Note:
        - The input tensors must have consistent feature dimensions for fusion to work correctly.
    """

    def __init__(
        self, feature_dims: List[int], reduce_fn: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self.reduce_fn = reduce_fn

        # All input dimensions must be the same
        self.output_dim = max(feature_dims)
        assert self.output_dim == min(feature_dims)

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the ReduceFusion module.

        Args:
            input_list (List[torch.Tensor]): A list of tensors to be fused.

        Returns:
            torch.Tensor: The fused tensor after applying the reduction function.
        """
        return self.reduce_fn(torch.stack(input_list))


def _verify_feature_dim(feature_dims: List[int]):
    """
    Verify that the feature dimensions in the list are valid.

    Args:
        feature_dims (List[int]): List of feature dimensions.

    Raises:
        AssertionError: If any feature dimension is non-positive or if the list is empty.
    """
    assert isinstance(feature_dims, list)
    assert all(x > 0 for x in feature_dims)
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
    Concatenates all inputs by their last dimension. The resulting tensor last dim will be
    the sum of the last dimension of all input tensors.
    """

    def __init__(self, feature_dims: List[int]):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self._output_dim = sum(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, sum(feature_dims)) where sum(feature_dims)
                is the sum of all input feature_dims.
        """
        return torch.cat(input_list, dim=-1)


class TemporalConcatFusion(nn.Module):
    """
    Concatenates all inputs by their temporal dimension which is assumed to be dim=1.
    """

    def __init__(self, feature_dims: List[int]):
        super().__init__()
        _verify_feature_dim(feature_dims)

        # All input dimensions must be the same
        self._output_dim = max(feature_dims)
        assert self._output_dim == min(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim)

        Returns:
            Tensor of shape (batch_size, sum(seq_len), feature_dim) where sum(seq_len) is
                the sum of all input tensors.
        """
        return torch.cat(input_list, dim=1)


class ReduceFusion(nn.Module):
    """
    Generic fusion method which takes a callable which takes the list of input tensors
    and expects a single tensor to be used. This class can be used to implement fusion
    methods like "sum", "max" and "prod".
    """

    def __init__(
        self, feature_dims: List[int], reduce_fn: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self.reduce_fn = reduce_fn

        # All input dimensions must be the same
        self._output_dim = max(feature_dims)
        assert self._output_dim == min(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, feature_dim).
        """
        return self.reduce_fn(torch.stack(input_list))


def _verify_feature_dim(feature_dims: List[int]):
    assert isinstance(feature_dims, list)
    assert all(x > 0 for x in feature_dims)

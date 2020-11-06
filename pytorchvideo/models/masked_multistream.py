# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
from torch import nn


"""
This file contains nn.Modules that take a tensor and mask in their forward function.
These masks can be used to represent invalid values (e.g. for tensors with varying
temporal dimension size). To easily compose these modules together, a
MaskedSequential module is provided.

Example usage:

    feature_dim = 64
    input_stream = MaskedSequential(
        PositionalEncoding(feature_dim),
        Dropout(p=0.1),
        TransposeMultiheadAttention(feature_dim),
        MaskedTemporalPooling(feature_dim, method="avg"),
        LayerNorm(feature_dim),
        LearnMaskedDefault(feature_dim),
    )

    input_tensor = ... # tensor with shape (batch_size, seq_len, feature_dim)
    mask_tensor = ... # bool tensor with shape (batch_size, seq_len)
    result = input_stream(input=input_tensor, mask=mask_tensor)
"""


class MaskedTemporalPooling(torch.nn.Module):
    """
    Applies temporal pooling operations on masked inputs. For each pooling operation
    all masked values are ignored.
    """

    def __init__(self, method: str):
        """
        method (str): the method of pooling to use. Options:
            - 'max': reduces temporal dimension to each valid max value.
            - 'avg': averages valid values in the temporal dimension.
            - 'sum': sums valid values in the temporal dimension.
            Note if all batch row elements are invalid, the temporal dimension is
            pooled to 0 values.
        """
        super().__init__()
        assert method in ("max", "avg", "sum")
        self._method = method

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        assert x.dim() == 3, "Requires x shape (batch_size x seq_len x feature_dim)"
        b, t = mask.shape[0], mask.shape[1]
        if self._method == "max":
            x[~mask, :] = float("-inf")

            # Invalid batch rows are set to 0.
            invalid_first_dim = ~mask.view(b, -1).any(dim=-1)
            x[invalid_first_dim, :] = 0

            x = torch.max(x, dim=1)[0]
        elif self._method == "avg":
            x = x * mask.unsqueeze(-1).float()
            mask = mask.view(b, t, -1).any(dim=-1)
            valid_lengths = mask.float().sum(dim=-1).int()
            x = x.sum(dim=1)
            x = x.div(valid_lengths.clamp(min=1).unsqueeze(-1).expand(x.size()).float())
        elif self._method == "sum":  # sum
            x = x * mask.unsqueeze(-1).float()
            x = x.sum(dim=1)
        else:
            raise NotImplementedError(
                f"{self._method} not available options are: 'max', 'avg', 'sum'"
            )

        return x


class TransposeMultiheadAttention(nn.Module):
    """
    Wrapper for nn.MultiheadAttention which first transposes the input tensor
    from (batch_size, seq_len, feature_dim) to (seq_length, batch_size, feature_dim),
    then applies the attention and transposes the attention outputs back to the input
    shape.
    """

    def __init__(self, feature_dim: int, num_heads: int = 1):
        """
        Args:
            feature_dim (int): attention embedding dimension
            num_heads (int): number of attention heads
        """
        super().__init__()
        self._attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads
        )
        self._attention_weights = None

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """
        Contains attention weights from last forward call.
        """
        return self._attention_weights

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, seq_len, feature_dim)
        """
        assert x.dim() == 3, "Requires x shape (batch_size x seq_len x feature_dim)"

        # At least the first element of each masked batch row must be valid for
        # key_padding_mask.
        mask[:, 0] = True

        # Transpose x to (seq_length x batch_size x feature_dim).
        x = x.transpose(0, 1)
        attn_output, self._attention_weights = self._attention(
            x, x, x, key_padding_mask=~mask
        )

        # Transpose attention output to (batch_size x seq_length x feature_dim).
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class LearnMaskedDefault(nn.Module):
    """
    Learns default values to fill invalid entries within input tensors. The
    invalid entries are represented by a mask which is passed into forward alongside
    the input tensor. Note the default value is only used if all entries in the batch row are
    invalid rather than just a portion of invalid entries within each batch row.
    """

    def __init__(
        self, feature_dim: int, init_method: str = "gaussian", freeze: bool = False
    ):
        """
        Args:
            feature_dim (int): the size of the default value parameter, this must match the
                input tensor size.
            init_method (str): the initial default value parameter. Options:
                - 'guassian'
                - 'zeros'
            freeze (bool): If True, the learned default parameter weights are frozen.
        """
        super().__init__()
        if init_method == "zeros":
            self._learned_defaults = nn.Parameter(
                torch.zeros(feature_dim), requires_grad=(not freeze)
            )
        elif init_method == "gaussian":
            self._learned_defaults = nn.Parameter(
                torch.Tensor(feature_dim), requires_grad=(not freeze)
            )
            nn.init.normal_(self._learned_defaults)
        else:
            raise NotImplementedError(
                f"{init_method} not available. Options are: 'zeros' or 'gaussian'"
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, feature_dim).
            mask (torch.Tensor): bool tensor of shape (batch_size, seq_len) If all elements
                in the batch dimension are False the learned default parameter is used for
                that batch element.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        # Determine which rows have no valid entries and use these for the default value mask.
        mask = mask.view(mask.shape[0], -1).any(dim=-1)
        for i in range(1, x.dim()):
            mask = mask.unsqueeze(i)
        x = x * mask.float() + self._learned_defaults * (1 - mask.float())
        return x


class MaskedSequential(nn.Sequential):
    """
    A sequential container that overrides forward to take a mask as well as the usual
    input tensor. This mask is only applied to modules in _MASK_MODULES (which take
    the mask argument).
    """

    _MASK_MODULES = [
        MaskedTemporalPooling,
        LearnMaskedDefault,
        TransposeMultiheadAttention,
    ]

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for module in self:
            if any(isinstance(module, mask_type) for mask_type in self._MASK_MODULES):
                input = module(input, mask=mask)
            else:
                input = module(input)

        return input

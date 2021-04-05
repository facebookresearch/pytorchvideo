# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytorchvideo.layers.utils import set_attributes
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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
            'max': reduces temporal dimension to each valid max value.
            'avg': averages valid values in the temporal dimension.
            'sum': sums valid values in the temporal dimension.
            Note if all batch row elements are invalid, the temporal dimension is
            pooled to 0 values.
        """
        super().__init__()
        assert method in ("max", "avg", "sum")
        self._method = method

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        assert x.dim() == 3, "Requires x shape (batch_size x seq_len x feature_dim)"
        b, t = x.shape[0], x.shape[1]
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool)

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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, seq_len, feature_dim)
        """
        assert x.dim() == 3, "Requires x shape (batch_size x seq_len x feature_dim)"

        if mask is not None:
            # At least the first element of each masked batch row must be valid for
            # key_padding_mask.
            mask[:, 0] = True
            mask = ~mask

        # Transpose x to (seq_length x batch_size x feature_dim).
        x = x.transpose(0, 1)
        attn_output, self._attention_weights = self._attention(
            x, x, x, key_padding_mask=mask
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
                'guassian'
                'zeros'
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


class LSTM(nn.Module):
    """
    Wrapper for torch.nn.LSTM that handles masked inputs.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Args:
          dim_in (int): input feature dimension
          hidden_dim (int): hidden dimesion of lstm layer
          dropout (float): dropout rate - 0.0 if no dropout
          bidirectional (bool): bidirectional or forward only
        """
        super().__init__()
        self.lstm = nn.LSTM(
            dim_in,
            hidden_dim,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.lstm.flatten_parameters()
        self.output_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.bidirectional = bidirectional

    def forward(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, output_dim) - outoput_dim is determined by
                hidden_dim and whether bidirectional or not
        """
        assert data.dim() == 3
        b, t = data.shape[0], data.shape[1]

        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool)

        lengths = mask.sum(axis=1)
        x_packed = pack_padded_sequence(
            data,
            lengths.clamp(1, data.size(1)),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h, _) = self.lstm(x_packed)

        if self.bidirectional:
            out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        else:
            out = h[-1, :, :]

        return out


class TransposeTransformerEncoder(nn.Module):
    """
    Wrapper for torch.nn.TransformerEncoder that handles masked inputs.
    """

    def __init__(
        self,
        dim_in: int,
        num_heads: int = 1,
        num_layers: int = 1,
    ):
        """
        Args:
          dim_in (int): input feature dimension
          num_heads (int): number of heads in the nn.MultiHeadAttention layers
          num_layers (int): the number of sub-encoder-layers in the encoder
        """
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads), num_layers
        )

    def forward(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        if mask is not None:
            # At least the first element of each masked batch row must be valid for
            # key_padding_mask.
            mask[:, 0] = True
            mask = ~mask

        out = self.encoder(
            src=data.transpose(0, 1), src_key_padding_mask=mask
        ).transpose(0, 1)

        return out[:, 0, :]


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
        LSTM,
        TransposeTransformerEncoder,
    ]

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for module in self:
            if any(isinstance(module, mask_type) for mask_type in self._MASK_MODULES):
                input = module(input, mask=mask)
            else:
                input = module(input)

        return input


class MaskedMultiPathWay(nn.Module):
    """
    Masked multi-pathway is composed of a list of stream nn.Modules followed by a
    fusion nn.Module that reduces these streams. Each stream module takes a mask
    and input tensor.

    ::

                            Pathway 1  ... Pathway N
                                ↓              ↓
                             Block 1        Block N
                                ↓⭠ --Fusion----↓
    """

    def __init__(
        self,
        *,
        multipathway_blocks: nn.ModuleList,
        multipathway_fusion: Optional[nn.Module],
    ) -> None:
        """
        Args:
            multipathway_blocks (nn.module_list): list of models from all pathways.
            multipathway_fusion (nn.module): fusion model.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(
        self, x_and_mask: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        out = []
        for pathway_idx in range(len(self.multipathway_blocks)):
            out.append(self.multipathway_blocks[pathway_idx](*x_and_mask[pathway_idx]))

        if self.multipathway_fusion is not None:
            x = self.multipathway_fusion(out)
        return x

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Applies a positional encoding to a tensor with shape (batch_size x seq_len x embed_dim).

    The positional encoding is computed as follows:
        PE(pos,2i) = sin(pos/10000^(2i/dmodel))
        PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

        where pos = position, pos in [0, seq_len)
        dmodel = data embedding dimension = embed_dim
        i = dimension index, i in [0, embed_dim)

    Reference: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
    Implementation Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embed_dim: int, seq_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim, dtype=torch.float)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pe.size(1) >= x.size(1), (
            "Cannot apply position encoding of size "
            + f"{self.pe.size()} when input has size {x.size()}"
        )
        return x + self.pe[:, : x.size(1), :]


class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x

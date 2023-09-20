# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
from torch import nn


class ScriptableSpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply spatiotemporal encoding to a tensor.

    This module is used for positional encoding in spatiotemporal models.
    It adds positional embeddings to the input tensor, which can be separated
    into spatial and temporal components or kept as a single positional encoding.

    Args:
        embed_dim (int): The embedding dimension for the input sequence.
        patch_embed_shape (Tuple[int, int, int]): The number of patches in each dimension
            (T, H, W) after patch embedding.
        sep_pos_embed (bool): If set to True, separate positional encodings are used for
            spatial patches and temporal sequences. Otherwise, a single positional encoding
            is used for all patches.
        has_cls (bool): If set to True, a cls token is added to the beginning of each
            input sequence.

    Note:
        - `patch_embed_shape` should be provided as a tuple in the form (T, H, W).
        - When `sep_pos_embed` is set to True, two positional encodings are used: one for
          spatial patches and one for temporal sequences. Otherwise, only one positional
          encoding is used for all patches.
        - If `has_cls` is set to True, a cls token is added to the beginning of each input
          sequence.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        
        super().__init__()
        assert len(patch_embed_shape) == 3, "Patch_embed_shape should be in the form of (T, H, W)."
        assert not has_cls  # This implementation currently does not support cls token.
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        # Initialize spatial and temporal positional embeddings.
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_spatial_patch, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patch, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatiotemporal positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with positional encoding applied.
        """
        B, N, C = x.shape

        assert self.sep_pos_embed
        pos_embed = self.pos_embed_spatial.repeat(1, self.num_temporal_patch, 1) + \
                    torch.repeat_interleave(self.pos_embed_temporal, self.num_spatial_patch, dim=1)

        x = x + pos_embed
        return x
    
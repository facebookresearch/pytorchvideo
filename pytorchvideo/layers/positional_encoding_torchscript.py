# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
from torch import nn


class ScriptableSpatioTemporalClsPositionalEncoding(nn.Module):
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
        assert not has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_spatial_patch, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_temporal_patch, embed_dim)
        )

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape

        assert self.sep_pos_embed
        pos_embed = self.pos_embed_spatial.repeat(
            1, self.num_temporal_patch, 1
        ) + torch.repeat_interleave(
            self.pos_embed_temporal,
            self.num_spatial_patch,
            dim=1,
        )
        x = x + pos_embed
        return x

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.layers.utils import set_attributes


class MemoryBank(nn.Module):
    """
    Performs Non-Parametric Instance Discrimination for self supervised learning on
    video. A memory bank is built to keep and update the historical feature embedding
    and use them for contrastive learning.

    The original paper is:
    Unsupervised Feature Learning via Non-Parametric Instance Discrimination
    https://arxiv.org/pdf/1805.01978.pdf

    More details can be found from the memory bank part in the following paper:
    Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/pdf/1911.05722.pdf
    """

    def __init__(
        self,
        backbone: nn.Module,
        mlp: Optional[nn.Module] = None,
        neg_size: int = 4096,
        temperature: float = 0.07,
        bank_size: int = 1280000,
        dim: int = 2048,
        mmt: float = 0.999,
    ) -> None:
        """
        Args:
            backbone (nn.Module): backbone used to forward the input.
            mlp (nn.Module): multi-layer perception used in memory bank instance
                discrimination model.
            neg_size (int): size of negative samples per instance.
            temperature (float): temperature to use for contrastive learning.
            bank_size (int): size of the memory bank, expected to be the same size as
                the training set.
            dim (int): dimension of the channel.
            mmt (float): momentum to use.
        """
        super().__init__()
        set_attributes(self, locals())
        self._init_mem_bank(bank_size, dim)

    def _init_mem_bank(self, bank_size: int, dim: int) -> None:
        """
        Given the memory bank size and the channel dimension, initialize the memory
            bank.
        Args:
            bank_size (int): size of the memory bank, expected to be the same size as
                 the training set.
            dim (int): dimension of the channel.
        """
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer(
            "memory",
            torch.rand(
                bank_size,
                dim,
            )
            .mul_(2 * stdv)
            .add_(-stdv)
            .to(next(self.backbone.parameters()).device),
        )

    def forward(self, x: torch.Tensor, x_ind: torch.Tensor) -> torch.Tensor:
        """
        Perform contrastive learning with random sampled negative instance from the
            memory bank. During training, update the memory bank with latest feature
            embedding.
        Args:
            x (torch.tensor): a batch of image with augmentation. The input tensor
                shape should able to be feed into the backbone.
            x_ind (torch.tensor): the index of the image x from the dataset. Expected
                shape is B.
        """
        batch_size = x.shape[0]
        x = self.backbone(x)
        if self.mlp is not None:
            x = self.mlp(x)
        # Normalize the output embedding before multiplication.
        x = F.normalize(x, p=2, dim=1)
        # Random sample negative instances from the memory bank.
        idx = torch.randint(0, self.bank_size, size=(batch_size, self.neg_size + 1)).to(
            x.device
        )
        # Fill the first with positive instances.
        idx.select(1, 0).copy_(x_ind.data)
        weight = torch.index_select(self.memory, 0, idx.view(-1)).detach()
        weight = weight.view(batch_size, self.neg_size + 1, self.dim)
        # Multiplication for contrastive learning.
        out = torch.einsum("bkc,bc->bk", weight, x)
        out = torch.div(out, self.temperature)
        gt = torch.zeros((batch_size,), device=x.device, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(out, gt)
        # Update memory during training.
        if self.training:
            with torch.no_grad():
                pos = torch.index_select(self.memory, 0, x_ind.view(-1))
                pos.mul_(self.mmt)
                pos.add_(torch.mul(x, 1 - self.mmt))
                norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated = pos.div(norm)
                self.memory.index_copy_(0, x_ind, updated)
        return loss

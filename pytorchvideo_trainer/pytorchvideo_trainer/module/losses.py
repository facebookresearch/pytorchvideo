# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List

import pytorchvideo_trainer.module.distributed_utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.layers.utils import set_attributes


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction (str): specifies reduction to apply to the output.
                It can be "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class NtxentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR Self-Supervised learning approach -
    https://arxiv.org/abs/2002.05709

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(
        self,
        temperature: float,
    ) -> None:
        super().__init__()
        set_attributes(self, locals())  # pyre-ignore[6]

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_list (list[torch.tensor]): A list of two tensors of shape N x C.
                Where, N is the batch size and C is the SSL model's embedding size.
        """
        assert (
            len(x_list) == 2
        ), f"Invalid list input to SimCLR. Expected dimention 2 but received {len(x_list)}"

        out_1, out_2 = x_list

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1 = du.AllGatherWithGradient.apply(out_1)  # pyre-ignore[16]
            out_2 = du.AllGatherWithGradient.apply(out_2)
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(out.shape[0], device=sim_matrix.device)
        ).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss


class SimilarityLoss(nn.Module):
    """
    Temperature-scaled Similarity loss for BYOL Self-Supervised learning
    approach - https://arxiv.org/abs/2006.07733

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q and k (nn.tensor): inputs to calculate the similarity, expected to have
                the same shape of `N x C`. Where N is the batch size and C
                is the SSL model's embedding size.
        """
        similarity = torch.einsum("nc,nc->n", [q, k])
        similarity /= self.temperature
        loss = -similarity.mean()
        return loss


class ContrastiveLoss(nn.Module):
    """
    Temperature-scaled Contrastive loss for MoCo and other Self-Supervised learning
    approaches - https://arxiv.org/abs/1911.05722

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(self, reduction: str = "mean", temperature: float = 0.1) -> None:
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (nn.tensor):  Expected to have the same shape of `N x C`.
                Where, N is the batch size and C is the SSL model's embedding size.
        """
        inputs = torch.div(inputs, self.temperature)
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).to(inputs.device)
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(inputs, targets)
        return loss

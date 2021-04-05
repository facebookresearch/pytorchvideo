# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
    Details can be found in:
    https://arxiv.org/pdf/2006.07733.pdf
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        feature_dim: int = 2048,
        predictor_inner: int = 4096,
        mmt: float = 0.99,
        norm: Callable = nn.SyncBatchNorm,
    ) -> None:
        """
        Args:
            backbone (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): stand projector is a mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            predictor (nn.Module): predictor MLP of BYOL of similar structure as the
                projector MLP.
            feature_dim (int): output feature dimension.
            predictor_inner (int): inner channel size for predictor.
            mmt (float): momentum update ratio for the momentum backbone.
            norm (callable): normalization to be used in projector, default is
                synchronized batchnorm.
        """
        super().__init__()
        self.mmt = mmt
        self.feature_dim = feature_dim
        if projector is not None:
            backbone = nn.Sequential(
                backbone,
                projector,
            )
        self.backbone = backbone
        self.backbone_mmt = copy.deepcopy(backbone)
        for p in self.backbone_mmt.parameters():
            p.requires_grad = False
        if predictor is None:
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, predictor_inner, bias=False),
                norm(predictor_inner),
                nn.ReLU(inplace=True),
                nn.Linear(predictor_inner, feature_dim, bias=True),
            )
        else:
            self.predictor = predictor

    def sim_loss(self, q, k):
        """
        Similarity loss for byol.
        Args:
            q and k (nn.tensor): inputs to calculate the similarity, expected to have
                the same shape of `N x C`.
        """
        similarity = torch.einsum("nc,nc->n", [q, k])
        loss = -similarity.mean()
        return loss

    def update_mmt(self, mmt: float):
        """
        Update the momentum. This function can be used to perform momentum annealing.
        Args:
            mmt (float): update the momentum.
        """
        self.mmt = mmt

    def get_mmt(self) -> float:
        """
        Get the momentum. This function can be used to perform momentum annealing.
        """
        return self.mmt

    @torch.no_grad()
    def _momentum_update_backbone(self):
        """
        Momentum update on the backbone.
        """
        for param, param_mmt in zip(
            self.backbone.parameters(), self.backbone_mmt.parameters()
        ):
            param_mmt.data = param_mmt.data * self.mmt + param.data * (1.0 - self.mmt)

    @torch.no_grad()
    def forward_backbone_mmt(self, x):
        """
        Forward momentum backbone.
        Args:
            x (tensor): input to be forwarded.
        """
        with torch.no_grad():
            proj = self.backbone_mmt(x)
        return F.normalize(proj, dim=1)

    def forward_backbone(self, x):
        """
        Forward backbone.
        Args:
            x (tensor): input to be forwarded.
        """
        proj = self.backbone(x)
        pred = self.predictor(proj)
        return F.normalize(pred, dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.tensor): a batch of image with augmentation. The input tensor
                shape should able to be feed into the backbone.
            x2 (torch.tensor): the size batch of image with different augmentation. The
                input tensor shape should able to be feed into the backbone.
        """
        pred_1 = self.forward_backbone(x1)
        pred_2 = self.forward_backbone(x2)

        with torch.no_grad():
            self._momentum_update_backbone()
            proj_mmt_1 = self.forward_backbone_mmt(x1)
            proj_mmt_2 = self.forward_backbone_mmt(x2)

        loss = (
            self.sim_loss(pred_1, proj_mmt_2) + self.sim_loss(pred_2, proj_mmt_1)
        ) / 2
        return loss

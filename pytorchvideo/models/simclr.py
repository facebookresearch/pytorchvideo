# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.distributed import differentiable_all_gather
from pytorchvideo.layers.utils import set_attributes


class SimCLR(nn.Module):
    """
    A Simple Framework for Contrastive Learning of Visual Representations
    Details can be found from:
    https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        mlp: nn.Module,
        backbone: Optional[nn.Module] = None,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.tensor): a batch of image with augmentation. The input tensor
                shape should able to be feed into the backbone.
            x2 (torch.tensor): the size batch of image with different augmentation. The
                input tensor shape should able to be feed into the backbone.
        """
        if self.backbone is not None:
            x1 = self.backbone(x1)
        x1 = self.mlp(x1)
        x1 = F.normalize(x1, p=2, dim=1)

        if self.backbone is not None:
            x2 = self.backbone(x2)
        x2 = self.mlp(x2)
        x2 = F.normalize(x2, p=2, dim=1)
        x2 = torch.cat(differentiable_all_gather(x2), dim=0)

        prod = torch.einsum("nc,kc->nk", [x1, x2])
        prod = prod.div(self.temperature)
        batch_size = x1.size(0)
        if dist.is_available() and dist.is_initialized():
            device_ind = dist.get_rank()
        else:
            device_ind = 0
        gt = (
            torch.tensor(
                list(range(device_ind * batch_size, (device_ind + 1) * batch_size))
            )
            .long()
            .to(x1.device)
        )
        loss = torch.nn.functional.cross_entropy(prod, gt)
        return loss

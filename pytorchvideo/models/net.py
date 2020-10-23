from typing import List, Optional

import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill
from pytorchvideo.models.utils import set_attributes


class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.

                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓

    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(self, *, blocks: nn.ModuleList) -> None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        init_net_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        return x


def init_net_weights(model: nn.Module, fc_init_std: float = 0.01) -> None:
    """
    Performs ResNet style weight initialization. That is, recursively initialize the
    given model in the following way for each type:
        Conv - Follow the initialization of kaiming_normal:
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
        BatchNorm - Set weight and bias of last BatchNorm at every residual bottleneck
            to 0.
        Linear - Set weight to 0 mean Gaussian with std deviation fc_init_std and bias
            to 0.
    Args:
        fc_init_std (float): the expected standard deviation for fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            if m.weight is not None:
                if hasattr(m, "block_final_bn") and m.block_final_bn:
                    m.weight.data.fill_(0.0)
                else:
                    m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()
    return model


class MultiPathWayWithFuse(nn.Module):
    """
    Build multi-pathway block with fusion for video recognition, each of the pathway
    contains its own Blocks and Fusion layers across different pathways.

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

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        for pathway_idx in range(len(self.multipathway_blocks)):
            if self.multipathway_blocks[pathway_idx] is not None:
                x[pathway_idx] = self.multipathway_blocks[pathway_idx](x[pathway_idx])
        if self.multipathway_fusion is not None:
            x = self.multipathway_fusion(x)
        return x

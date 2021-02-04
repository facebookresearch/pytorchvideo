# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill


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
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
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
            if m.bias is not None:
                m.bias.data.zero_()
    return model

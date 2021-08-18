# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill
from pytorchvideo.layers import SpatioTemporalClsPositionalEncoding


def _init_resnet_weights(model: nn.Module, fc_init_std: float = 0.01) -> None:
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
        model (nn.Module): Model to be initialized.
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


def _init_vit_weights(model: nn.Module, trunc_normal_std: float = 0.02) -> None:
    """
    Weight initialization for vision transformers.

    Args:
        model (nn.Module): Model to be initialized.
        trunc_normal_std (float): the expected standard deviation for fully-connected
            layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)


def init_net_weights(
    model: nn.Module,
    init_std: float = 0.01,
    style: str = "resnet",
) -> None:
    """
    Performs weight initialization. Options include ResNet style weight initialization
    and transformer style weight initialization.

    Args:
        model (nn.Module): Model to be initialized.
        init_std (float): The expected standard deviation for initialization.
        style (str): Options include "resnet" and "vit".
    """
    assert style in ["resnet", "vit"]
    if style == "resnet":
        return _init_resnet_weights(model, init_std)
    elif style == "vit":
        return _init_vit_weights(model, init_std)
    else:
        raise NotImplementedError

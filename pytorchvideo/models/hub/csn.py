# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.csn import create_csn
from torch.hub import load_state_dict_from_url

"""
Channel-Separated Convolutional Network models for video recognition.
"""

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
checkpoint_paths = {
    "csn_r101": f"{root_dir}/CSN_32x2_R101.pyth",
}


def csn_r101(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Channel-Separated Convolutional Networks (CSN) R101 model architecture [1]
    with pretrained weights based on 32x2 setting on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 77.0 (trained on 16x8 GPUs).

    [1] "Video classification with channel-separated convolutional networks"
        Du Tran, Heng Wang, Lorenzo Torresani, Matt Feiszli. ICCV 2019.
        https://arxiv.org/abs/1904.02811

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    model = create_csn(
        model_depth=101,
        stem_pool=nn.MaxPool3d,
        head_pool_kernel_size=(4, 7, 7),
        **kwargs,
    )

    if pretrained:
        path = checkpoint_paths["csn_r101"]
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)

    return model

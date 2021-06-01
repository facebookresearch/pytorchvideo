# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.r2plus1d import create_r2plus1d
from torch.hub import load_state_dict_from_url

"""
R(2+1)D style models for video recognition.
"""

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
checkpoint_paths = {
    "r2plus1d_r50": f"{root_dir}/R2PLUS1D_16x4_R50.pyth",
}


def r2plus1d_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""

    R(2+1)D model architecture from [1] with pretrained weights based on 16x4 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 76.01.
    (trained on 8*8 GPUs)

    [1] "A closer look at spatiotemporal convolutions for action recognition"
        Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri. CVPR 2018.
        https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    model = create_r2plus1d(dropout_rate=0.5, **kwargs)

    if pretrained:
        path = checkpoint_paths["r2plus1d_r50"]
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)

    return model

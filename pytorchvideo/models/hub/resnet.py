# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet
from torch.hub import load_state_dict_from_url


"""
ResNet style models for video recognition.
"""

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
checkpoint_paths = {
    "slow_r50": f"{root_dir}/SLOW_8x8_R50.pyth",
    "c2d_r50": f"{root_dir}/C2D_8x8_R50.pyth",
    "i3d_r50": f"{root_dir}/I3D_8x8_R50.pyth",
}


def _resnet(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: str = "",
    **kwargs: Any,
) -> nn.Module:
    model = create_resnet(**kwargs)
    if pretrained:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model


def slow_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Slow R50 model architecture [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 74.58.

    [1] "SlowFast Networks for Video Recognition"
        Christoph Feichtenhofer et al
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slow_r50"],
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(8, 7, 7),
        model_depth=50,
        **kwargs,
    )


def c2d_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    C2D R50 model architecture with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 71.46.

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["c2d_r50"],
        stem_conv_kernel_size=(1, 7, 7),
        stage1_pool=nn.MaxPool3d,
        stage_conv_a_kernel_size=(
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        **kwargs,
    )


def i3d_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    I3D R50 model architecture from [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 73.27.

    [1] "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/abs/1705.07750

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["i3d_r50"],
        stem_conv_kernel_size=(5, 7, 7),
        stage1_pool=nn.MaxPool3d,
        stage_conv_a_kernel_size=(
            (3, 1, 1),
            [(3, 1, 1), (1, 1, 1)],
            [(3, 1, 1), (1, 1, 1)],
            [(1, 1, 1), (3, 1, 1)],
        ),
        **kwargs,
    )

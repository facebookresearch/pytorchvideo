# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast
from torch.hub import load_state_dict_from_url


root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
checkpoint_paths = {
    "slowfast_r50": f"{root_dir}/SLOWFAST_8x8_R50.pyth",
    "slowfast_r101": f"{root_dir}/SLOWFAST_8x8_R101.pyth",
    "slowfast_16x8_r101_50_50": f"{root_dir}/SLOWFAST_16x8_R101_50_50.pyth",
}


def _slowfast(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: str = "",
    **kwargs: Any,
) -> nn.Module:
    model = create_slowfast(**kwargs)
    if pretrained:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model


def slowfast_r50(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R50 model architecture [1] trained with an 8x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 76.4.

    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_r50"],
        model_depth=50,
        slowfast_fusion_conv_kernel_size=(7, 1, 1),
        **kwargs,
    )


def slowfast_r101(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R101 model architecture [1] trained with an 8x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 77.9.

    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_r101"],
        model_depth=101,
        slowfast_fusion_conv_kernel_size=(5, 1, 1),
        **kwargs,
    )


def slowfast_16x8_r101_50_50(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R101_50_50 model architecture [1] trained with an 16x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 78.7.

    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    # slowfast_r101_50_50 has 6 conv blocks with kernel=(3, 1, 1) in stage 4.
    stage_conv_a_kernel_sizes = (
        (
            (1, 1, 1),
            (1, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
        (
            (3, 1, 1),
            (3, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
    )
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_16x8_r101_50_50"],
        model_depth=101,
        slowfast_fusion_conv_kernel_size=(5, 1, 1),
        stage_conv_a_kernel_sizes=stage_conv_a_kernel_sizes,
        head_pool_kernel_sizes=((16, 7, 7), (64, 7, 7)),
        **kwargs,
    )

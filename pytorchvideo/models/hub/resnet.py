from typing import Any

import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet
from torch.hub import load_state_dict_from_url


"""
ResNet style models for video recognition.
"""

checkpoint_paths = {
    "slow_r50": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth"
}


def slow_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Slow R50 model architecture [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 74.58.

    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
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
    model = create_resnet(
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(8, 7, 7),
        model_depth=50,
        **kwargs,
    )

    if pretrained:
        path = checkpoint_paths["slow_r50"]
        checkpoint = load_state_dict_from_url(path, progress=progress)
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)

    return model

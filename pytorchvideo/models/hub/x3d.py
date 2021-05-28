# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Optional

import torch.nn as nn
from pytorchvideo.models.x3d import create_x3d
from torch.hub import load_state_dict_from_url


root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
checkpoint_paths = {
    "x3d_xs": f"{root_dir}/X3D_XS.pyth",
    "x3d_s": f"{root_dir}/X3D_S.pyth",
    "x3d_m": f"{root_dir}/X3D_M.pyth",
    "x3d_l": f"{root_dir}/X3D_L.pyth",
}


def _x3d(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    model = create_x3d(**kwargs)
    if pretrained and checkpoint_path is not None:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model


def x3d_xs(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
):
    r"""
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 69.12.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_xs"],
        input_clip_length=4,
        input_crop_size=160,
        **kwargs,
    )


def x3d_s(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 73.33.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_s"],
        input_clip_length=13,
        input_crop_size=160,
        **kwargs,
    )


def x3d_m(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 75.94.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_m"],
        input_clip_length=16,
        input_crop_size=224,
        **kwargs,
    )


def x3d_l(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
):
    """
    X3D-XS model architecture [1] trained on the Kinetics dataset.
    Model with pretrained weights has top1 accuracy of 77.44.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/x3d.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["x3d_l"],
        input_clip_length=16,
        input_crop_size=312,
        depth_factor=5.0,
        **kwargs,
    )

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Optional

import torch.nn as nn
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import create_x3d
from torch.hub import load_state_dict_from_url


_root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
_checkpoint_paths = {
    "efficient_x3d_xs": f"{_root_dir}/efficient_x3d_xs_original_form.pyth",
    "efficient_x3d_s": f"{_root_dir}/efficient_x3d_s_original_form.pyth",
}


def _efficient_x3d(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: Optional[str] = None,
    # Model params
    expansion: str = "XS",
    **kwargs: Any,
) -> nn.Module:

    model = create_x3d(
        expansion=expansion,
        **kwargs,
    )

    if pretrained and checkpoint_path is not None:
        # All models are loaded onto CPU by default
        state_dict = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        model.load_state_dict(state_dict)

    return model


def efficient_x3d_xs(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""
    X3D-XS model architectures [1] with pretrained weights trained
    on the Kinetics dataset with efficient implementation for mobile cpu.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetcis-400 dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        To modify any other model settings, specify them in the kwargs.
        All the args are defined in pytorchvideo/models/x3d.py
    """
    return _efficient_x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=_checkpoint_paths["efficient_x3d_xs"],
        expansion="XS",
        **kwargs,
    )


def efficient_x3d_s(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""
    X3D-S model architectures [1] with pretrained weights trained
    on the Kinetics dataset with efficient implementation for mobile cpu.

    [1] Christoph Feichtenhofer, "X3D: Expanding Architectures for
    Efficient Video Recognition." https://arxiv.org/abs/2004.04730

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetcis-400 dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        To modify any other model settings, specify them in the kwargs.
        All the args are defined in pytorchvideo/models/x3d.py
    """
    return _efficient_x3d(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=_checkpoint_paths["efficient_x3d_s"],
        expansion="S",
        **kwargs,
    )

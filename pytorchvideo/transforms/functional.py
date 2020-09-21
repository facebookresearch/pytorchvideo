# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math

import torch


def uniform_temporal_subsample(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.

    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        num_samples (int): The number of equispaced samples to be selected

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32

    _, t, _, _ = x.shape
    assert num_samples <= t
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, 1, indices)


def short_side_scale(
    x: torch.Tensor, size: int, interpolation: str = "bilinear"
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.

    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'

    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    _, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation)

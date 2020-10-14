# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Tuple

import torch


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = 1
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples <= t
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


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
    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)


def repeat_temporal_frames_subsample(
    frames: torch.Tensor, frame_ratios: Tuple[int], temporal_dim: int = 1
) -> Tuple[torch.Tensor]:
    """
    Prepare output as a list of tensors subsampled from the input frames. Each tensor
        maintain a unique copy of subsampled frames, which corresponds to a unique
        pathway.
    Args:
        frames (tensor): frames of images sampled from the video. Expected to have
            torch tensor (including int, long, float, complex, etc) with dimension
            larger than one.
        frame_ratios (tuple): ratio to perform temporal down-sampling for each pathways.
        temporal_dim (int): dimension of temporal.
    Returns:
        frame_list (tuple): list of tensors as output.
    """
    temporal_length = frames.shape[temporal_dim]
    frame_list = []
    for ratio in frame_ratios:
        pathway = uniform_temporal_subsample(
            frames, temporal_length // ratio, temporal_dim
        )
        frame_list.append(pathway)
    return frame_list

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Tuple

import cv2
import numpy as np
import torch


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = 1
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


@torch.jit.ignore
def _interpolate_opencv(
    x: torch.Tensor, size: Tuple[int, int], interpolation: str
) -> torch.Tensor:
    """
    Down/up samples the input torch tensor x to the given size with given interpolation
    mode.
    Args:
        input (Tensor): the input tensor to be down/up sampled.
        size (Tuple[int, int]): expected output spatial size.
        interpolation: model to perform interpolation, options include `nearest`,
            `linear`, `bilinear`, `bicubic`.
    """
    _opencv_pytorch_interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_AREA,
        "bicubic": cv2.INTER_CUBIC,
    }
    assert interpolation in _opencv_pytorch_interpolation_map
    new_h, new_w = size
    img_array_list = [
        img_tensor.squeeze(0).numpy()
        for img_tensor in x.permute(1, 2, 3, 0).split(1, dim=0)
    ]
    resized_img_array_list = [
        cv2.resize(
            img_array,
            (new_w, new_h),  # The input order for OpenCV is w, h.
            interpolation=_opencv_pytorch_interpolation_map[interpolation],
        )
        for img_array in img_array_list
    ]
    img_array = np.concatenate(
        [np.expand_dims(img_array, axis=0) for img_array in resized_img_array_list],
        axis=0,
    )
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_array))
    img_tensor = img_tensor.permute(3, 0, 1, 2)
    return img_tensor


def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
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
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181

    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ("pytorch", "opencv")
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == "pytorch":
        return torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=False
        )
    elif backend == "opencv":
        return _interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
    else:
        raise NotImplementedError(f"{backend} backend not supported.")


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


def convert_to_one_hot(targets: torch.Tensor, num_class: int) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors, given the number of classes.
    """

    assert (
        torch.max(targets).item() < num_class
    ), "Class Index must be less than number of classes"

    one_hot_targets = torch.zeros(
        (targets.shape[0], num_class), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)

    return one_hot_targets


def uniform_crop(frames: torch.Tensor, size: int, spatial_idx: int = 1) -> torch.Tensor:
    """
    Perform uniform spatial sampling on the frames based on three-crop setting.
        If width is larger than height, take left, center and right crop.
        If height is larger than width, take top, center, and bottom crop.
    Args:
        frames (tensor): A video tensor of shape (C, T, H, W) to perform uniform crop.
        size (int): Desired height and weight size to crop the frames.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
    Returns:
        cropped (tensor): A cropped video tensor of shape (C, T, size, size).
    """

    assert spatial_idx in [0, 1, 2]
    height = frames.shape[2]
    width = frames.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = frames[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    return cropped

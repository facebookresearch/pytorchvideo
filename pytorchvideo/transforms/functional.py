# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import math
from typing import Tuple

import numpy as np
import torch


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
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
    if not _HAS_CV2:
        raise ImportError(
            "opencv is required to use opencv transforms. Please "
            "install with 'pip install opencv-python'."
        )

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


def uniform_temporal_subsample_repeated(
    frames: torch.Tensor, frame_ratios: Tuple[int], temporal_dim: int = -3
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


def convert_to_one_hot(
    targets: torch.Tensor,
    num_class: int,
    label_smooth: float = 0.0,
) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    Args:
        targets (torch.Tensor): Index labels to be converted.
        num_class (int): Total number of classes.
        label_smooth (float): Label smooth value for non-target classes. Label smooth
            is disabled by default (0).
    """
    assert (
        torch.max(targets).item() < num_class
    ), "Class Index must be less than number of classes"
    assert 0 <= label_smooth < 1.0, "Label smooth value needs to be between 0 and 1."

    non_target_value = label_smooth / num_class
    target_value = 1.0 - label_smooth + non_target_value
    one_hot_targets = torch.full(
        (targets.shape[0], num_class),
        non_target_value,
        dtype=torch.long if label_smooth == 0.0 else None,
        device=targets.device,
    )
    one_hot_targets.scatter_(1, targets.long().view(-1, 1), target_value)
    return one_hot_targets


def short_side_scale_with_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `channel` x `num frames` x `height` x `width`.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        (tensor): the scaled images with dimension of
            `channel` x `num frames` x `height` x `width`.
        (tensor): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    c, t, h, w = images.shape
    images = short_side_scale(images, size, interpolation, backend)
    _, _, new_h, new_w = images.shape
    if w < h:
        boxes *= float(new_h) / h
    else:
        boxes *= float(new_w) / w
    return images, boxes


def random_short_side_scale_with_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    min_size: int,
    max_size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `channel` x `num frames` x `height` x `width`.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        (tensor): the scaled images with dimension of
            `channel` x `num frames` x `height` x `width`.
        (tensor): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    size = torch.randint(min_size, max_size + 1, (1,)).item()
    return short_side_scale_with_boxes(images, boxes, size, interpolation, backend)


def random_crop_with_boxes(
    images: torch.Tensor, size: int, boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `channel` x `num frames` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): cropped images with dimension of
            `channel` x `num frames` x `height` x `width`.
        cropped_boxes (tensor): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    cropped_boxes = crop_boxes(boxes, x_offset, y_offset)
    return cropped, clip_boxes_to_image(
        cropped_boxes, cropped.shape[-2], cropped.shape[-1]
    )


def _uniform_crop_helper(images: torch.Tensor, size: int, spatial_idx: int):
    """
    A helper function grouping the common components in uniform crop
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

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
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    return cropped, x_offset, y_offset


def uniform_crop(
    images: torch.Tensor,
    size: int,
    spatial_idx: int,
) -> torch.Tensor:
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `channel` x `num frames` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
    Returns:
        cropped (tensor): images with dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    cropped, _, _ = _uniform_crop_helper(images, size, spatial_idx)
    return cropped


def uniform_crop_with_boxes(
    images: torch.Tensor,
    size: int,
    spatial_idx: int,
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `channel` x `num frames` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): images with dimension of
            `channel` x `num frames` x `height` x `width`.
        cropped_boxes (tensor): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped, x_offset, y_offset = _uniform_crop_helper(images, size, spatial_idx)
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset)
    return cropped, clip_boxes_to_image(
        cropped_boxes, cropped.shape[-2], cropped.shape[-1]
    )


def horizontal_flip_with_boxes(
    prob: float, images: torch.Tensor, boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `channel` x `num frames` x `height` x `width`.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `channel` x `num frames` x `height` x `width`.
        flipped_boxes (tensor): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    flipped_boxes = copy.deepcopy(boxes)

    if np.random.uniform() < prob:
        images = images.flip((-1))
        width = images.shape[3]
        flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (tensor): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (tensor): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = copy.deepcopy(boxes)
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def crop_boxes(boxes: torch.Tensor, x_offset: int, y_offset: int) -> torch.Tensor:
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (torch.Tensor): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (torch.Tensor): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = copy.deepcopy(boxes)
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes

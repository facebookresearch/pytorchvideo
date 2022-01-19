# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
pulished under an Apache License 2.0.
COMMENT FROM ORIGINAL:
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
from typing import Optional, Tuple

import torch


def _get_pixels(
    per_pixel: bool,
    rand_color: bool,
    patch_size: Tuple[int],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    A utility function that generates image patches for RandomErasing transform
    """
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """
    This variant of RandomErasing is intended to be applied to a video tensor i.e,
    batch of images after it has been normalized by dataset mean and std.

    Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf

    Args:
        probability (float): Probability that the Random Erasing operation will be performed.
        min_area (float): Minimum percentage of erased area wrt input image area.
        max_area (float): Maximum percentage of erased area wrt input image area.
        min_aspect (float): Minimum aspect ratio of erased area.
        mode (str): pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count (int): maximum number of erasing blocks per image, area per box is scaled by
            count. Per-image count is randomly chosen between 1 and this value.
        min_count (int): minimum number of erasing blocks per image, area per box is scaled by
            count. Per-image count is randomly chosen between 1 and this value.
        device (str): Device to perform the transform on.
    """

    def __init__(
        self,
        probability: float = 0.5,
        min_area: float = 0.02,
        max_area: float = 1 / 3,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
        mode: str = "const",
        min_count: int = 1,
        max_count: Optional[int] = None,
        num_splits: int = 0,
        device: str = "cuda",
        cube: bool = True,
    ) -> None:
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio: Tuple[float, float] = (
            math.log(min_aspect),
            math.log(max_aspect),
        )
        self.min_count = min_count
        self.max_count: int = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color: bool = False
        self.per_pixel: bool = False
        self.cube = cube
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device

    def _erase(
        self, img: torch.Tensor, chan: int, height: int, width: int, dtype: torch.dtype
    ) -> None:
        if random.random() > self.probability:
            return
        area = height * width
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < width and h < height:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),  # pyre-ignore[6]
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def _erase_cube(
        self,
        video: torch.Tensor,
        batch_start: int,
        batch_size: int,
        chan: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        if random.random() > self.probability:
            return
        area = height * width
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(100):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < width and h < height:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)
                    for i in range(batch_start, batch_size):
                        img_instance = video[i]
                        img_instance[:, top : top + h, left : left + w] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),  # pyre-ignore[6]
                            dtype=dtype,
                            device=self.device,
                        )
                    break

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        """
        # Expects frames of shape T, C, H, W
        batch_size, chan, height, width = frames.size()
        # skip first slice of batch if num_splits is set (for clean portion of samples)
        batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
        if self.cube:
            self._erase_cube(
                frames,
                batch_start,
                batch_size,
                chan,
                height,
                width,
                frames.dtype,
            )
        else:
            for i in range(batch_start, batch_size):
                self._erase(frames[i], chan, height, width, frames.dtype)
        return frames

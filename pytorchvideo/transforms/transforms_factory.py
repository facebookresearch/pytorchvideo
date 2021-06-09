# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ConvertUint8ToFloat,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)


def create_video_transform(
    mode: str,
    video_key: Optional[str] = None,
    remove_key: Optional[List[str]] = None,
    num_samples: Optional[int] = 8,
    convert_to_float: bool = True,
    video_mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
    video_std: Tuple[float, float, float] = (0.225, 0.225, 0.225),
    min_size: int = 256,
    max_size: int = 320,
    crop_size: Union[int, Tuple[int, int]] = 224,
    horizontal_flip_prob: float = 0.5,
) -> Union[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
]:
    """
    Function that returns a factory default callable video transform, with default
    parameters that can be modified. The transform that is returned depends on the
    ``mode`` parameter: when in "train" mode, we use randomized transformations,
    and when in "val" mode, we use the corresponding deterministic transformations.
    Depending on whether ``video_key`` is set, the input to the transform can either
    be a video tensor or a dict containing ``video_key`` that maps to a video
    tensor. The video tensor should be of shape (C, T, H, W).

              "train" mode                         "val" mode

       (UniformTemporalSubsample)          (UniformTemporalSubsample)
                   ↓                                   ↓
         (ConvertUint8ToFloat)               (ConvertUint8ToFloat)
                   ↓                                   ↓
               Normalize                           Normalize
                   ↓                                   ↓
          RandomShortSideScale                   ShortSideScale
                   ↓                                   ↓
               RandomCrop                          CenterCrop
                   ↓
          RandomHorizontalFlip

    (transform) = transform can be included or excluded in the returned
                  composition of transformations

    Args:
        mode (str): 'train' or 'val'. We use randomized transformations in
            'train' mode, and we use the corresponding deterministic transformation
            in 'val' mode.
        video_key (str, optional): Optional key for video value in dictionary input.
            When video_key is None, the input is assumed to be a torch.Tensor.
            Default is None.
        remove_key (List[str], optional): Optional key to remove from a dictionary input.
            Default is None.
        num_samples (int, optional): The number of equispaced samples to be selected in
            UniformTemporalSubsample. If None, then UniformTemporalSubsample will not be
            used. Default is 8.
        convert_to_float (bool): If True, converts images from uint8 to float.
            Otherwise, leaves the image as is. Default is True.
        video_mean (Tuple[float, float, float]): Sequence of means for each channel to
            normalize to zero mean and unit variance. Default is (0.45, 0.45, 0.45).
        video_std (Tuple[float, float, float]): Sequence of standard deviations for each
            channel to normalize to zero mean and unit variance.
            Default is (0.225, 0.225, 0.225).
        min_size (int): Minimum size that the shorter side is scaled to for
            RandomShortSideScale. If in "val" mode, this is the exact size
            the the shorter side is scaled to for ShortSideScale.
            Default is 256.
        max_size (int): Maximum size that the shorter side is scaled to for
            RandomShortSideScale. Default is 340.
        crop_size (int or Tuple[int, int]): Desired output size of the crop for RandomCrop
            in "train" mode and CenterCrop in "val" mode. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made. Default is 224.
        horizontal_flip_prob (float): Probability of the video being flipped in
            RandomHorizontalFlip. Default value is 0.5.

    Returns:
        A factory-default callable composition of transforms.
    """
    if isinstance(crop_size, int):
        assert crop_size <= min_size, "crop_size must be less than or equal to min_size"
    elif isinstance(crop_size, tuple):
        assert (
            max(crop_size) <= min_size
        ), "the height and width in crop_size must be less than or equal to min_size"
    else:
        raise TypeError
    if video_key is None:
        assert remove_key is None, "remove_key should be None if video_key is None"

    transform = Compose(
        (
            []
            if num_samples is None
            else [UniformTemporalSubsample(num_samples=num_samples)]
        )
        + ([ConvertUint8ToFloat()] if convert_to_float else [])
        + [Normalize(mean=video_mean, std=video_std)]
        + (
            [
                RandomShortSideScale(
                    min_size=min_size,
                    max_size=max_size,
                ),
                RandomCrop(size=crop_size),
                RandomHorizontalFlip(p=horizontal_flip_prob),
            ]
            if mode == "train"
            else [
                ShortSideScale(size=min_size),
                CenterCrop(size=crop_size),
            ]
        )
    )

    if video_key is None:
        return transform

    return Compose(
        [
            ApplyTransformToKey(
                key=video_key,
                transform=transform,
            )
        ]
        + ([] if remove_key is None else [RemoveKey(k) for k in remove_key])
    )

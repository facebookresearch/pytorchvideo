# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    AugMix,
    ConvertUint8ToFloat,
    Normalize,
    RandomResizedCrop,
    Permute,
    RandomShortSideScale,
    RandAugment,
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


_RANDAUG_DEFAULT_PARAS = {
    "magnitude": 9,
    "num_layers": 2,
    "prob": 0.5,
    "transform_hparas": None,
    "sampling_type": "gaussian",
    "sampling_hparas": None,
}

_AUGMIX_DEFAULT_PARAS = {
    "magnitude": 3,
    "alpha": 1.0,
    "width": 3,
    "depth": -1,
    "transform_hparas": None,
    "sampling_hparas": None,
}

_RANDOM_RESIZED_CROP_DEFAULT_PARAS = {
    "scale": (0.08, 1.0),
    "aspect_ratio": (3.0 / 4.0, 4.0 / 3.0),
}


def _get_augmentation(
    aug_type: str, aug_paras: Optional[Dict[str, Any]] = None
) -> List[Callable]:
    """
    Initializes a list of callable transforms for video augmentation.

    Args:
        aug_type (str): Currently supports 'default', 'randaug', or 'augmix'.
            Returns an empty list when aug_type is 'default'. Returns a list
            of transforms containing RandAugment when aug_type is 'randaug'
            and a list containing AugMix when aug_type is 'augmix'.
        aug_paras (Dict[str, Any], optional): A dictionary that contains the necessary
            parameters for the augmentation set in aug_type. If any parameters are
            missing or if None, default parameters will be used. Default is None.

    Returns:
        aug (List[Callable]): List of callable transforms with the specified augmentation.
    """

    if aug_paras is None:
        aug_paras = {}

    if aug_type == "default":
        aug = []
    elif aug_type == "randaug":
        aug = [
            Permute((1, 0, 2, 3)),
            RandAugment(
                magnitude=aug_paras.get(
                    "magnitude", _RANDAUG_DEFAULT_PARAS["magnitude"]
                ),
                num_layers=aug_paras.get(
                    "num_layers", _RANDAUG_DEFAULT_PARAS["num_layers"]
                ),
                prob=aug_paras.get("prob", _RANDAUG_DEFAULT_PARAS["prob"]),
                sampling_type=aug_paras.get(
                    "sampling_type", _RANDAUG_DEFAULT_PARAS["sampling_type"]
                ),
                sampling_hparas=aug_paras.get(
                    "sampling_hparas", _RANDAUG_DEFAULT_PARAS["sampling_hparas"]
                ),
            ),
            Permute((1, 0, 2, 3)),
        ]
    elif aug_type == "augmix":
        aug = [
            Permute((1, 0, 2, 3)),
            AugMix(
                magnitude=aug_paras.get(
                    "magnitude", _AUGMIX_DEFAULT_PARAS["magnitude"]
                ),
                alpha=aug_paras.get("alpha", _AUGMIX_DEFAULT_PARAS["alpha"]),
                width=aug_paras.get("width", _AUGMIX_DEFAULT_PARAS["width"]),
                depth=aug_paras.get("depth", _AUGMIX_DEFAULT_PARAS["depth"]),
            ),
            Permute((1, 0, 2, 3)),
        ]
    else:
        raise NotImplementedError

    return aug


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
    aug_type: str = "default",
    aug_paras: Optional[Dict[str, Any]] = None,
    random_resized_crop_paras: Optional[Dict[str, Any]] = None,
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

                       "train" mode                                 "val" mode

                (UniformTemporalSubsample)                  (UniformTemporalSubsample)
                            ↓
                   (RandAugment/AugMix)                                 ↓
                            ↓
                  (ConvertUint8ToFloat)                       (ConvertUint8ToFloat)
                            ↓                                           ↓
                        Normalize                                   Normalize
                            ↓                                           ↓
    RandomResizedCrop/RandomShortSideScale+RandomCrop       ShortSideScale+CenterCrop
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
        aug_type (str): Currently supports 'default', 'randaug', or 'augmix'. No
            augmentations other than RandomShortSideScale and RandomCrop area performed
            when aug_type is 'default'. RandAugment is used when aug_type is 'randaug'
            and AugMix is used when aug_type is 'augmix'. Default is 'default'.
        aug_paras (Dict[str, Any], optional): A dictionary that contains the necessary
            parameters for the augmentation set in aug_type. If any parameters are
            missing or if None, default parameters will be used. Default is None.
        random_resized_crop_paras (Dict[str, Any], optional): A dictionary that contains
            the necessary parameters for Inception-style cropping. This crops the given
            videos to random size and aspect ratio. A crop of random size relative to the
            original size and a random aspect ratio is made. This crop is finally resized
            to given size. This is popularly used to train the Inception networks. If any
            parameters are missing or if None, default parameters in
            _RANDOM_RESIZED_CROP_DEFAULT_PARAS will be used. If None, RandomShortSideScale
            and RandomCrop will be used as a fallback. Default is None.

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
    if aug_type == "default":
        assert aug_paras is None, "aug_paras should be None for ``default`` aug_type"

    if random_resized_crop_paras is not None:
        random_resized_crop_paras["target_height"] = crop_size
        random_resized_crop_paras["target_width"] = crop_size
        if "scale" not in random_resized_crop_paras:
            random_resized_crop_paras["scale"] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS[
                "scale"
            ]
        if "aspect_ratio" not in random_resized_crop_paras:
            random_resized_crop_paras[
                "aspect_ratio"
            ] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS["aspect_ratio"]

    transform = Compose(
        (
            []
            if num_samples is None
            else [UniformTemporalSubsample(num_samples=num_samples)]
        )
        + (
            _get_augmentation(aug_type=aug_type, aug_paras=aug_paras)
            if mode == "train"
            else []
        )
        + ([ConvertUint8ToFloat()] if convert_to_float else [])
        + [Normalize(mean=video_mean, std=video_std)]
        + (
            (
                [RandomResizedCrop(**random_resized_crop_paras)]
                if random_resized_crop_paras is not None
                else [
                    RandomShortSideScale(
                        min_size=min_size,
                        max_size=max_size,
                    ),
                    RandomCrop(size=crop_size),
                ]
                + [RandomHorizontalFlip(p=horizontal_flip_prob)]
            )
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

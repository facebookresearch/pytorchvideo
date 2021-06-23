# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, Optional

import torch
from pytorchvideo.transforms.augmentations import AugmentTransform
from pytorchvideo.transforms.augmentations import (
    _decreasing_int_to_arg,
    _decreasing_to_arg,
    _increasing_randomly_negate_to_arg,
    _increasing_magnitude_to_arg,
    _AUGMENTATION_MAX_LEVEL,
)
from pytorchvideo.transforms.transforms import OpSampler


_AUGMIX_LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": _increasing_randomly_negate_to_arg,
    "Posterize": _decreasing_int_to_arg,
    "Solarize": _decreasing_to_arg,
    "ShearX": _increasing_randomly_negate_to_arg,
    "ShearY": _increasing_randomly_negate_to_arg,
    "TranslateX": _increasing_randomly_negate_to_arg,
    "TranslateY": _increasing_randomly_negate_to_arg,
    "AdjustSaturation": _increasing_magnitude_to_arg,
    "AdjustContrast": _increasing_magnitude_to_arg,
    "AdjustBrightness": _increasing_magnitude_to_arg,
    "AdjustSharpness": _increasing_magnitude_to_arg,
}

_TRANSFORM_AUGMIX_MAX_PARAMS = {
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": (0, 30),
    "Posterize": (4, 4),
    "Solarize": (1, 1),
    "ShearX": (0, 0.3),
    "ShearY": (0, 0.3),
    "TranslateX": (0, 1.0 / 3.0),
    "TranslateY": (0, 1.0 / 3.0),
    "AdjustSaturation": (0.1, 1.8),
    "AdjustContrast": (0.1, 1.8),
    "AdjustBrightness": (0.1, 1.8),
    "AdjustSharpness": (0.1, 1.8),
}

# Hyperparameters for sampling magnitude.
# sampling_data_type determines whether uniform sampling samples among ints or floats.
# sampling_min determines the minimum possible value obtained from uniform
# sampling among floats.
SAMPLING_AUGMIX_DEFAULT_HPARAS = {"sampling_data_type": "float", "sampling_min": 0.1}


class AugMix:
    """
    This implements AugMix for video. AugMix generates several chains of augmentations
    on the original video, which are then mixed together with each other and with the
    original video to create an augmented video. The input video tensor should have
    shape (T, C, H, W).

    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
    (https://arxiv.org/pdf/1912.02781.pdf)
    """

    def __init__(
        self,
        magnitude: int = 3,
        alpha: float = 1.0,
        width: int = 3,
        depth: int = -1,
        transform_hparas: Optional[Dict[str, Any]] = None,
        sampling_hparas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            magnitude (int): Magnitude used for transform function. Default is 3.
            alpha (float): Parameter for choosing mixing weights from the beta
                and Dirichlet distributions. Default is 1.0.
            width (int): The number of transformation chains. Default is 3.
            depth (int): The number of transformations in each chain. If depth is -1,
                each chain will have a random length between 1 and 3 inclusive.
                Default is -1.
            transform_hparas (Optional[Dict[Any]]): Transform hyper parameters.
                Needs to have key fill. By default, the fill value is (0.5, 0.5, 0.5).
            sampling_hparas (Optional[Dict[Any]]): Hyper parameters for sampling. If
                gaussian sampling is used, it needs to have key sampling_std. By
                default, it uses SAMPLING_AUGMIX_DEFAULT_HPARAS.
        """
        assert isinstance(magnitude, int), "magnitude must be an int"
        assert (
            magnitude >= 1 and magnitude <= _AUGMENTATION_MAX_LEVEL
        ), f"magnitude must be between 1 and {_AUGMENTATION_MAX_LEVEL} inclusive"
        assert alpha > 0.0, "alpha must be greater than 0"
        assert width > 0, "width must be greater than 0"

        self._magnitude = magnitude

        self.dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([alpha] * width)
        )
        self.beta = torch.distributions.beta.Beta(alpha, alpha)

        transforms_list = [
            AugmentTransform(
                transform_name=transform_name,
                magnitude=self._magnitude,
                prob=1.0,
                level_to_arg=_AUGMIX_LEVEL_TO_ARG,
                transform_max_paras=_TRANSFORM_AUGMIX_MAX_PARAMS,
                transform_hparas=transform_hparas,
                sampling_type="uniform",
                sampling_hparas=sampling_hparas or SAMPLING_AUGMIX_DEFAULT_HPARAS,
            )
            for transform_name in list(_TRANSFORM_AUGMIX_MAX_PARAMS.keys())
        ]
        if depth > 0:
            self.augmix_fn = OpSampler(
                transforms_list,
                num_sample_op=depth,
                replacement=True,
            )
        else:
            self.augmix_fn = OpSampler(
                transforms_list,
                num_sample_op=3,
                randomly_sample_depth=True,
                replacement=True,
            )

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Perform AugMix to the input video tensor.

        Args:
            video (torch.Tensor): Input video tensor with shape (T, C, H, W).
        """
        mixing_weights = self.dirichlet.sample()
        m = self.beta.sample().item()
        mixed = torch.zeros(video.shape, dtype=torch.float32)
        for mw in mixing_weights:
            mixed += mw * self.augmix_fn(video)
        if video.dtype == torch.uint8:
            return (m * video + (1 - m) * mixed).type(torch.uint8)
        else:
            return m * video + (1 - m) * mixed

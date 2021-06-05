# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, Optional

import torch
from pytorchvideo.transforms.augmentations import AugmentTransform
from pytorchvideo.transforms.transforms import OpSampler

# A dictionary that contains transform names (key) and their corresponding maximum
# transform magnitude (value).
_TRANSFORM_RANDAUG_MAX_PARAMS = {
    "AdjustBrightness": (1, 0.9),
    "AdjustContrast": (1, 0.9),
    "AdjustSaturation": (1, 0.9),
    "AdjustSharpness": (1, 0.9),
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": (0, 30),
    "Posterize": (4, 4),
    "Solarize": (1, 1),
    "ShearX": (0, 0.3),
    "ShearY": (0, 0.3),
    "TranslateX": (0, 0.45),
    "TranslateY": (0, 0.45),
}

# Hyperparameters for sampling magnitude.
SAMPLING_RANDAUG_DEFAULT_HPARAS = {"sampling_std": 0.5}


class RandAugment:
    """
    This implements RandAugment for video. Assume the input video tensor with shape
    (T, C, H, W).

    RandAugment: Practical automated data augmentation with a reduced search space
    (https://arxiv.org/abs/1909.13719)
    """

    def __init__(
        self,
        magnitude: int = 9,
        num_layers: int = 2,
        prob: float = 0.5,
        sampling_type: str = "gaussian",
        sampling_hparas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        This implements RandAugment for video.

        Args:
            magnitude (int): Magnitude used for transform function.
            num_layers (int): How many transform functions to apply for each
                augmentation.
            prob (float): The probablity of applying each transform function.
            sampling_type (str): Sampling method for magnitude of transform. It should
                be either gaussian or uniform.
            sampling_hparas (Optional[Dict[Any]]): Hyper parameters for sampling. If
                gaussian sampling is used, it needs to have key sampling_std. By
                default, it uses transform_default_hparas.
        """
        assert sampling_type in ["gaussian", "uniform"]
        sampling_hparas = sampling_hparas or SAMPLING_RANDAUG_DEFAULT_HPARAS
        if sampling_type == "gaussian":
            assert "sampling_std" in sampling_hparas

        randaug_fn = [
            AugmentTransform(
                transform_name,
                magnitude,
                prob=prob,
                transform_max_paras=_TRANSFORM_RANDAUG_MAX_PARAMS,
                sampling_type=sampling_type,
                sampling_hparas=sampling_hparas,
            )
            for transform_name in list(_TRANSFORM_RANDAUG_MAX_PARAMS.keys())
        ]
        self.randaug_fn = OpSampler(randaug_fn, num_sample_op=num_layers)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Perform RandAugment to the input video tensor.

        Args:
            video (torch.Tensor): Input video tensor with shape (T, C, H, W).
        """
        return self.randaug_fn(video)

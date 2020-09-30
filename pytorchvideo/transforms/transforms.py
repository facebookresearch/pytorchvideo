# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict

import pytorchvideo.transforms.functional
import torch


class MapDict:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.MapDict(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class UniformTemporalSubsample(torch.nn.Module):
    """
    nn.Module wrapper for pytorchvideo.transforms.functional.uniform_temporal_subsample.
    """

    def __init__(self, num_samples: int):
        super().__init__()
        self._num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pytorchvideo.transforms.functional.uniform_temporal_subsample(
            x, self._num_samples
        )


class RandomShortSideScale(torch.nn.Module):
    """
    nn.Module wrapper for pytorchvideo.transforms.functional.short_side_scale. The size
    parameter is chosen randomly between min_size and max_size.
    """

    def __init__(self, min_size: int, max_size: int):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = torch.randint(self._min_size, self._max_size, (1,)).item()
        return pytorchvideo.transforms.functional.short_side_scale(x, size)
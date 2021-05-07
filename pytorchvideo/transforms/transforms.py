# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Tuple

import pytorchvideo.transforms.functional
import torch
import torchvision.transforms


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
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


class RemoveKey(torch.nn.Module):
    """
    Removes the given key from the input dict. Useful for removing modalities from a
    video clip that aren't needed.
    """

    def __init__(self, key: str):
        self._key = key

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        if self._key in x:
            del x[self._key]
        return x


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int):
        super().__init__()
        self._num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.uniform_temporal_subsample(
            x, self._num_samples
        )


class UniformTemporalSubsampleRepeated(torch.nn.Module):
    """
    ``nn.Module`` wrapper for
    ``pytorchvideo.transforms.functional.uniform_temporal_subsample_repeated``.
    """

    def __init__(self, frame_ratios: Tuple[int]):
        super().__init__()
        self._frame_ratios = frame_ratios

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.uniform_temporal_subsample_repeated(
            x, self._frame_ratios
        )


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(self, size: int):
        super().__init__()
        self._size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.short_side_scale(x, self._size)


class RandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(self, min_size: int, max_size: int):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return pytorchvideo.transforms.functional.short_side_scale(x, size)


class UniformCropVideo(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_crop``.
    """

    def __init__(
        self, size: int, video_key: str = "video", aug_index_key: str = "aug_index"
    ):
        super().__init__()
        self._size = size
        self._video_key = video_key
        self._aug_index_key = aug_index_key

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        x[self._video_key] = pytorchvideo.transforms.functional.uniform_crop(
            x[self._video_key], self._size, x[self._aug_index_key]
        )
        return x


class Normalize(torchvision.transforms.Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation

    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid


class OpSampler(torch.nn.Module):
    """
    Given a list of transforms with weights, OpSampler applies weighted sampling to
    select n transforms, which are then applied sequentially to the input.
    """

    def __init__(
        self,
        transforms_list: List[Callable],
        transforms_prob: Optional[List[float]] = None,
        num_sample_op: int = 1,
        replacement: bool = False,
    ):
        """
        Args:
            transforms_list (List[Callable]): A list of tuples of all available transforms
                to sample from.
            transforms_prob (Optional[List[float]]): The probabilities associated with
                each transform in transforms_list. If not provided, the sampler assumes a
                uniform distribution over all transforms. They do not need to sum up to one
                but weights need to be positive.
            num_sample_op (int): Number of transforms to sample and apply to input.
            replacement (bool): If replacement is True, transforms are drawn with replacement.
        """
        super().__init__()
        assert len(transforms_list) > 0, "Argument transforms_list cannot be empty."
        assert num_sample_op > 0, "Need to sample at least one transform."
        assert num_sample_op <= len(
            transforms_list
        ), "Argument num_sample_op cannot be greater than number of available transforms."

        if transforms_prob is not None:
            assert len(transforms_list) == len(
                transforms_prob
            ), "Argument transforms_prob needs to have the same length as transforms_list."

            assert (
                min(transforms_prob) > 0
            ), "Argument transforms_prob needs to be greater than 0."

        self.transforms_list = transforms_list
        self.transforms_prob = torch.FloatTensor(
            transforms_prob
            if transforms_prob is not None
            else [1] * len(transforms_list)
        )
        self.num_sample_op = num_sample_op
        self.replacement = replacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        index_list = torch.multinomial(
            self.transforms_prob, self.num_sample_op, replacement=self.replacement
        )

        for index in index_list:
            x = self.transforms_list[index](x)

        return x

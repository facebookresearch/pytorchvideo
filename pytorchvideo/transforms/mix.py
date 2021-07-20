# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Tuple

import torch
from pytorchvideo.transforms.functional import convert_to_one_hot


def _mix_labels(
    labels: torch.Tensor,
    num_classes: int,
    lam: float = 1.0,
    label_smoothing: float = 0.0,
):
    """
    This function converts class indices to one-hot vectors and mix labels, given the
    number of classes.

    Args:
        labels (torch.Tensor): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixing labels.
        label_smoothing (float): Label smoothing value.
    """
    labels1 = convert_to_one_hot(labels, num_classes, label_smoothing)
    labels2 = convert_to_one_hot(labels.flip(0), num_classes, label_smoothing)
    return labels1 * lam + labels2 * (1.0 - lam)


class MixUp(torch.nn.Module):
    """
    Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        num_classes: int = 400,
    ) -> None:
        """
        This implements MixUp for videos.

        Args:
            alpha (float): Mixup alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.mixup_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        assert x.size(0) > 1, "MixUp cannot be applied to a single instance."

        mixup_lambda = self.mixup_beta_sampler.sample()
        x_flipped = x.flip(0).mul_(1.0 - mixup_lambda)
        x.mul_(mixup_lambda).add_(x_flipped)
        new_labels = _mix_labels(
            labels,
            self.num_classes,
            mixup_lambda,
            self.label_smoothing,
        )
        return x, new_labels


class CutMix(torch.nn.Module):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        num_classes: int = 400,
    ) -> None:
        """
        This implements CutMix for videos.

        Args:
            alpha (float): CutMix alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.cutmix_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _clip(self, value: int, min_value: int, max_value: int) -> int:
        """
        Clip value based on minimum value and maximum value.
        """

        return min(max(value, min_value), max_value)

    def _get_rand_box(self, input_shape: Tuple[int], cutmix_lamda: float) -> Tuple[int]:
        """
        Get a random square box given a lambda value.
        """

        ratio = (1 - cutmix_lamda) ** 0.5
        input_h, input_w = input_shape[-2:]
        cut_h, cut_w = int(input_h * ratio), int(input_w * ratio)
        cy = torch.randint(input_h, (1,)).item()
        cx = torch.randint(input_w, (1,)).item()
        yl = self._clip(cy - cut_h // 2, 0, input_h)
        yh = self._clip(cy + cut_h // 2, 0, input_h)
        xl = self._clip(cx - cut_w // 2, 0, input_w)
        xh = self._clip(cx + cut_w // 2, 0, input_w)
        return yl, yh, xl, xh

    def _cutmix(
        self, x: torch.Tensor, cutmix_lamda: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform CutMix and return corrected lambda value.
        """

        yl, yh, xl, xh = self._get_rand_box(x.size(), cutmix_lamda)
        box_area = float((yh - yl) * (xh - xl))
        cutmix_lamda_corrected = 1.0 - box_area / (x.size(-2) * x.size(-1))
        x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        return x, cutmix_lamda_corrected

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        assert x.size(0) > 1, "Cutmix cannot be applied to a single instance."
        assert x.dim() == 4 or x.dim() == 5, "Please correct input shape."

        cutmix_lamda = self.cutmix_beta_sampler.sample()
        x, cutmix_lamda_corrected = self._cutmix(x, cutmix_lamda)
        new_labels = _mix_labels(
            labels,
            self.num_classes,
            cutmix_lamda_corrected,
            self.label_smoothing,
        )
        return x, new_labels


class MixVideo(torch.nn.Module):
    """
    Stochastically applies either MixUp or CutMix to the input video.
    """

    def __init__(
        self,
        cutmix_prob: float = 0.5,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        label_smoothing: float = 0.0,
        num_classes: int = 400,
    ):
        """
        Args:
            cutmix_prob (float): Probability of using CutMix. MixUp will be used with
                probability 1 - cutmix_prob. If cutmix_prob is 0, then MixUp is always
                used. If cutmix_prob is 1, then CutMix is always used.
            mixup_alpha (float): MixUp alpha value.
            cutmix_alpha (float): CutMix alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """

        assert 0.0 <= cutmix_prob <= 1.0, "cutmix_prob should be between 0.0 and 1.0"

        super().__init__()
        self.cutmix_prob = cutmix_prob
        self.mixup = MixUp(
            alpha=mixup_alpha, label_smoothing=label_smoothing, num_classes=num_classes
        )
        self.cutmix = CutMix(
            alpha=cutmix_alpha, label_smoothing=label_smoothing, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """

        if torch.rand(1).item() < self.cutmix_prob:
            return self.cutmix(x, labels)
        else:
            return self.mixup(x, labels)

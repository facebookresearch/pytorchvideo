# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, Tuple

import torch
from pytorchvideo.transforms.functional import convert_to_one_hot


def _mix_labels(
    labels: torch.Tensor,
    num_classes: int,
    lam: float = 1.0,
    label_smoothing: float = 0.0,
    one_hot: bool = False,
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
    if one_hot:
        labels1 = labels
        labels2 = labels.flip(0)
    else:
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
        one_hot: bool = False,
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
        self.one_hot = one_hot

    def forward(
        self,
        x_video: torch.Tensor,
        labels: torch.Tensor,
        **args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
            Optional: x_audio: Audio input tensor.
        """
        assert x_video.size(0) > 1, "MixUp cannot be applied to a single instance."
        mixup_lambda = self.mixup_beta_sampler.sample()
        x_video_flipped = x_video.flip(0).mul_(1.0 - mixup_lambda)
        x_video.mul_(mixup_lambda).add_(x_video_flipped)

        new_labels = _mix_labels(
            labels,
            self.num_classes,
            mixup_lambda,
            self.label_smoothing,
            one_hot=self.one_hot,
        )

        if args.get("x_audio", None) is not None:
            x_audio = args["x_audio"]
            assert x_audio.size(0) > 1, "MixUp cannot be applied to a single instance."
            x_audio_flipped = x_audio.flip(0).mul_(1.0 - mixup_lambda)
            x_audio.mul_(mixup_lambda).add_(x_audio_flipped)
            return x_video, x_audio, new_labels
        else:
            return x_video, new_labels


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
        one_hot: bool = False,
    ) -> None:
        """
        This implements CutMix for videos.

        Args:
            alpha (float): CutMix alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.one_hot = one_hot
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
        self,
        x_video: torch.Tensor,
        labels: torch.Tensor,
        **args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        assert x_video.size(0) > 1, "Cutmix cannot be applied to a single instance."
        assert x_video.dim() == 4 or x_video.dim() == 5, "Please correct input shape."
        cutmix_lamda = self.cutmix_beta_sampler.sample()
        x_video, cutmix_lamda_corrected = self._cutmix(x_video, cutmix_lamda)
        new_labels = _mix_labels(
            labels,
            self.num_classes,
            cutmix_lamda_corrected,
            self.label_smoothing,
            one_hot=self.one_hot,
        )
        if args.get("x_audio", None) is not None:
            x_audio = args["x_audio"]
            assert x_audio.size(0) > 1, "Cutmix cannot be applied to a single instance."
            assert (
                x_audio.dim() == 4 or x_audio.dim() == 5
            ), "Please correct input shape."
            x_audio, _ = self._cutmix(x_audio, cutmix_lamda)
            return x_video, x_audio, new_labels
        else:
            return x_video, new_labels


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
        one_hot: bool = False,
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
            alpha=mixup_alpha,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
            one_hot=one_hot,
        )
        self.cutmix = CutMix(
            alpha=cutmix_alpha, label_smoothing=label_smoothing, num_classes=num_classes
        )

    # def forward(self, x: torch.Tensor, labels: torch.Tensor):
    def forward(
        self,
        x_video: torch.Tensor,
        labels: torch.Tensor,
        **args: Any,
    ) -> Dict[str, Any]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        if args.get("x_audio", None) is None:
            if torch.rand(1).item() < self.cutmix_prob:
                x_video, new_labels = self.cutmix(x_video, labels)
            else:
                x_video, new_labels = self.mixup(x_video, labels)
            return x_video, new_labels
        else:
            x_audio = args["x_audio"]
            if torch.rand(1).item() < self.cutmix_prob:
                x_video, new_labels, x_audio = self.cutmix(x_video, labels, x_audio)
            else:
                x_video, new_labels, x_audio = self.mixup(x_video, labels, x_audio)
            return x_video, x_audio, new_labels

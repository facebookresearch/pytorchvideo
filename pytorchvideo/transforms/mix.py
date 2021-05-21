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
            labels (torch.Tensor): Labels for input.
        """
        assert x.size(0) > 1, "MixUp cannot be applied to a single instance."

        mixup_lamda = self.mixup_beta_sampler.sample()
        x_mixed = x * mixup_lamda + x.flip(0) * (1.0 - mixup_lamda)
        new_labels = _mix_labels(
            labels,
            self.num_classes,
            mixup_lamda,
            self.label_smoothing,
        )
        return x_mixed, new_labels

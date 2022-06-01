# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.transforms.functional import convert_to_one_hot


class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Adapted from Classy Vision: ./classy_vision/losses/soft_target_cross_entropy_loss.py.
    This allows the targets for the cross entropy loss to be multi-label.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        normalize_targets: bool = True,
    ) -> None:
        """
        Args:
            ignore_index (int): sample should be ignored for loss if the class is this value.
            reduction (str): specifies reduction to apply to the output.
            normalize_targets (bool): whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample.
        """
        super().__init__()
        set_attributes(self, locals())
        assert isinstance(self.normalize_targets, bool)
        if self.reduction not in ["mean", "none"]:
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self.reduction)
            )
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): the shape of the tensor is N x C, where N is the number of
                samples and C is the number of classes. The tensor is raw input without
                softmax/sigmoid.
            target (torch.Tensor): the shape of the tensor is N x C or N. If the shape is N, we
                will convert the target to one hot vectors.
        """
        # Check if targets are inputted as class integers
        if target.ndim == 1:
            assert (
                input.shape[0] == target.shape[0]
            ), "SoftTargetCrossEntropyLoss requires input and target to have same batch size!"
            target = convert_to_one_hot(target.view(-1, 1), input.shape[1])

        assert input.shape == target.shape, (
            "SoftTargetCrossEntropyLoss requires input and target to be same "
            f"shape: {input.shape} != {target.shape}"
        )

        # Samples where the targets are ignore_index do not contribute to the loss
        N, C = target.shape
        valid_mask = torch.ones((N, 1), dtype=torch.float).to(input.device)
        if 0 <= self.ignore_index <= C - 1:
            drop_idx = target[:, self.ignore_idx] > 0
            valid_mask[drop_idx] = 0

        valid_targets = target.float() * valid_mask
        if self.normalize_targets:
            valid_targets /= self.eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(input, -1)

        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        # Perform reduction
        if self.reduction == "mean":
            # Normalize based on the number of samples with > 0 non-ignored targets
            loss = per_sample_loss.sum() / torch.sum(
                (torch.sum(valid_mask, -1) > 0)
            ).clamp(min=1)
        elif self.reduction == "none":
            loss = per_sample_loss

        return loss

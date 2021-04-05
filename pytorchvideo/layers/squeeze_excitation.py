# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Optional

import torch
import torch.nn as nn
from pytorchvideo.models.resnet import ResBlock


class SqueezeAndExcitationLayer2D(nn.Module):
    """2D Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(
        self,
        in_planes: int,
        reduction_ratio: Optional[int] = 16,
        reduced_planes: Optional[int] = None,
    ):

        """
        Args:
            in_planes (int): input channel dimension.
            reduction_ratio (int): factor by which in_planes should be reduced to
                get the output channel dimension.
            reduced_planes (int): Output channel dimension. Only one of reduction_ratio
                or reduced_planes should be defined.
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_planes is defined
        assert bool(reduction_ratio) != bool(
            reduced_planes
        ), "Only of reduction_ratio or reduced_planes should be defined for SE layer"

        reduced_planes = (
            in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): 2D image of format C * H * W
        """
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


def create_audio_2d_squeeze_excitation_block(
    dim_in: int,
    dim_out: int,
    use_se=False,
    se_reduction_ratio=16,
    branch_fusion: Callable = lambda x, y: x + y,
    # Conv configs.
    conv_a_kernel_size: int = 3,
    conv_a_stride: int = 1,
    conv_a_padding: int = 1,
    conv_b_kernel_size: int = 3,
    conv_b_stride: int = 1,
    conv_b_padding: int = 1,
    # Norm configs.
    norm: Callable = nn.BatchNorm2d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:

    """
    2-D Residual block with squeeze excitation (SE2D) for 2d. Performs a summation between an
    identity shortcut in branch1 and a main block in branch2. When the input and
    output dimensions are different, a convolution followed by a normalization
    will be performed.

    ::

                                         Input
                                           |-------+
                                           ↓       |
                                         conv2d    |
                                           ↓       |
                                          Norm     |
                                           ↓       |
                                       activation  |
                                           ↓       |
                                         conv2d    |
                                           ↓       |
                                          Norm     |
                                           ↓       |
                                          SE2D     |
                                           ↓       }
                                       Summation ←-+
                                           ↓
                                       Activation

    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).
    Transform examples include: BottleneckBlock.

    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_out (int): output channel size of the bottleneck.
        use_se (bool): if true, use squeeze excitation layer in the bottleneck.
        se_reduction_ratio (int): factor by which input channels should be reduced to
            get the output channel dimension in SE layer.
        branch_fusion (callable): a callable that constructs summation layer.
            Examples include: lambda x, y: x + y, OctaveSum.

        conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple): convolutional padding(s) for conv_a.
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.

        norm (callable): a callable that constructs normalization layer. Examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer in
            bottleneck and block. Examples include: nn.ReLU, nn.Softmax, nn.Sigmoid,
            and None (not performing activation).

    Returns:
        (nn.Module): resnet basic block layer.
    """

    branch2 = [
        nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=conv_a_kernel_size,
            stride=conv_a_stride,
            padding=conv_a_padding,
            bias=False,
        ),
        norm(dim_out, norm_eps, norm_momentum),
        activation() if activation else nn.Identity(),
        nn.Conv2d(
            dim_out,
            dim_out,
            kernel_size=conv_b_kernel_size,
            stride=conv_b_stride,
            padding=conv_b_padding,
            bias=False,
        ),
        norm(dim_out, norm_eps, norm_momentum),
    ]
    if use_se:
        branch2.append(
            SqueezeAndExcitationLayer2D(dim_out, reduction_ratio=se_reduction_ratio)
        )
    branch2 = nn.Sequential(*branch2)

    branch1_conv, branch1_norm = None, None
    if conv_a_stride * conv_b_stride != 1 or dim_in != dim_out:
        branch1_conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=conv_a_stride * conv_b_stride,
            bias=False,
        )
        branch1_norm = norm(dim_out, norm_eps, norm_momentum)

    return ResBlock(
        branch1_conv=branch1_conv,
        branch1_norm=branch1_norm,
        branch2=branch2,
        activation=activation() if activation else None,
        branch_fusion=branch_fusion,
    )

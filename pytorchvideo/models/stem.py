# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.convolutions import ConvReduce3D
from pytorchvideo.layers.utils import set_attributes


def create_res_basic_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (3, 7, 7),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (1, 3, 3),
    conv_bias: bool = False,
    conv: Callable = nn.Conv3d,
    # Pool configs.
    pool: Callable = nn.MaxPool3d,
    pool_kernel_size: Tuple[int] = (1, 3, 3),
    pool_stride: Tuple[int] = (1, 2, 2),
    pool_padding: Tuple[int] = (0, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the basic resnet stem layer. It performs spatiotemporal Convolution, BN, and
    Relu following by a spatiotemporal pooling.

    ::

                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    Normalization options include: BatchNorm3d and None (no normalization).
    Activation options include: ReLU, Softmax, Sigmoid, and None (no activation).
    Pool3d options include: AvgPool3d, MaxPool3d, and None (no pooling).

    Args:

        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.
        conv (callable): Callable used to build the convolution layer.

        pool (callable): a callable that constructs pooling layer, options include:
            nn.AvgPool3d, nn.MaxPool3d, and None (not performing pooling).
        pool_kernel_size (tuple): pooling kernel size(s).
        pool_stride (tuple): pooling stride size(s).
        pool_padding (tuple): pooling padding size(s).

        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): resnet basic stem layer.
    """
    conv_module = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,
    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()
    pool_module = (
        None
        if pool is None
        else pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
    )

    return ResNetBasicStem(
        conv=conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=pool_module,
    )


def create_acoustic_res_basic_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (3, 7, 7),
    conv_stride: Tuple[int] = (1, 1, 1),
    conv_padding: Tuple[int] = (1, 3, 3),
    conv_bias: bool = False,
    # Pool configs.
    pool: Callable = nn.MaxPool3d,
    pool_kernel_size: Tuple[int] = (1, 3, 3),
    pool_stride: Tuple[int] = (1, 2, 2),
    pool_padding: Tuple[int] = (0, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the acoustic resnet stem layer. It performs a spatial and a temporal
    Convolution in parallel, then performs, BN, and Relu following by a spatiotemporal
    pooling.

    ::

                                    Conv3d   Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    Normalization options include: BatchNorm3d and None (no normalization).
    Activation options include: ReLU, Softmax, Sigmoid, and None (no activation).
    Pool3d options include: AvgPool3d, MaxPool3d, and None (no pooling).

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s), it will be performed as
            temporal and spatial convolution in parallel.
        conv_padding (tuple): convolutional padding size(s), it  will be performed
            as temporal and spatial convolution in parallel.
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.

        pool (callable): a callable that constructs pooling layer, options include:
            nn.AvgPool3d, nn.MaxPool3d, and None (not performing pooling).
        pool_kernel_size (tuple): pooling kernel size(s).
        pool_stride (tuple): pooling stride size(s).
        pool_padding (tuple): pooling padding size(s).

        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): resnet basic stem layer.
    """
    conv_module = ConvReduce3D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(
            # Temporal conv kernel size.
            (conv_kernel_size[0], 1, 1),
            # Spatial conv kernel size.
            (1, conv_kernel_size[1], conv_kernel_size[2]),
        ),
        stride=(conv_stride, conv_stride),
        padding=((conv_padding[0], 0, 0), (0, conv_padding[1], conv_padding[2])),
        bias=(conv_bias, conv_bias),
        reduction_method="sum",
    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()
    pool_module = (
        None
        if pool is None
        else pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
    )

    return ResNetBasicStem(
        conv=conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=pool_module,
    )


class ResNetBasicStem(nn.Module):
    """
    ResNet basic 3D stem module. Performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.

    ::

                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    The builder can be found in `create_res_basic_stem`.
    """

    def __init__(
        self,
        *,
        conv: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        pool: nn.Module = None,
    ) -> None:
        """
        Args:
            conv (torch.nn.modules): convolutional module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

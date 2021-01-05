# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from torch.nn.common_types import _size_3_t


class ConvReduce3D(nn.Module):
    """
    Builds a list of convolutional operators and performs summation on the outputs.

                            Conv3d, Conv3d, ...,  Conv3d
                                           ↓
                                          Sum
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[_size_3_t],
        stride: Optional[Tuple[_size_3_t]] = None,
        padding: Optional[Tuple[_size_3_t]] = None,
        padding_mode: Optional[Tuple[str]] = None,
        dilation: Optional[Tuple[_size_3_t]] = None,
        groups: Optional[Tuple[int]] = None,
        bias: Optional[Tuple[bool]] = None,
        reduction_method: str = "sum",
    ) -> None:
        """
        Args:
            in_channels int: number of input channels.
            out_channels int: number of output channels produced by the convolution(s).
            kernel_size tuple(_size_3_t): Tuple of sizes of the convolutionaling kernels.
            stride tuple(_size_3_t): Tuple of strides of the convolutions.
            padding tuple(_size_3_t): Tuple of paddings added to all three sides of the
                input.
            padding_mode tuple(string): Tuple of padding modes for each convs.
                Options include `zeros`, `reflect`, `replicate` or `circular`.
            dilation tuple(_size_3_t): Tuple of spacings between kernel elements.
            groups tuple(_size_3_t): Tuple of numbers of blocked connections from input
                channels to output channels.
            bias tuple(bool): If `True`, adds a learnable bias to the output.
            reduction_method str: Options include `sum` and `cat`.
        """
        super().__init__()
        assert reduction_method in ("sum", "cat")
        self.reduction_method = reduction_method
        conv_list = []
        for ind in range(len(kernel_size)):
            conv_param = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size[ind],
            }
            if stride is not None and stride[ind] is not None:
                conv_param["stride"] = stride[ind]
            if padding is not None and padding[ind] is not None:
                conv_param["padding"] = padding[ind]
            if dilation is not None and dilation[ind] is not None:
                conv_param["dilation"] = dilation[ind]
            if groups is not None and groups[ind] is not None:
                conv_param["groups"] = groups[ind]
            if bias is not None and bias[ind] is not None:
                conv_param["bias"] = bias[ind]
            if padding_mode is not None and padding_mode[ind] is not None:
                conv_param["padding_mode"] = padding_mode[ind]
            conv_list.append(nn.Conv3d(**conv_param))
        self.convs = nn.ModuleList(conv_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        for ind in range(len(self.convs)):
            output.append(self.convs[ind](x))
        if self.reduction_method == "sum":
            output = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduction_method == "cat":
            output = torch.cat(output, dim=1)
        return output


def create_conv_2plus1d(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    inner_channels: int = None,
    conv_xy_first: bool = False,
    kernel_size: Tuple[int] = (3, 3, 3),
    stride: Tuple[int] = (2, 2, 2),
    padding: Tuple[int] = (1, 1, 1),
    bias: bool = False,
    dilation: Tuple[int] = (1, 1, 1),
    groups: int = 1,
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Create a 2plus1d conv layer. It performs spatiotemporal Convolution, BN, and
    Relu following by a spatiotemporal pooling.

                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)

    Normalization options include: BatchNorm3d and None (no normalization).
    Activation options include: ReLU, Softmax, Sigmoid, and None (no activation).

    Args:
        Convolution related configs:
            in_channels (int): input channel size of the convolution.
            out_channels (int): output channel size of the convolution.
            kernel_size (tuple): convolutional kernel size(s).
            stride (tuple): convolutional stride size(s).
            padding (tuple): convolutional padding size(s).
            bias (bool): convolutional bias. If true, adds a learnable bias to the
                output.
            groups (int): Number of groups in convolution layers. value >1 is unsupported.
            dilation (tuple): dilation value in convolution layers. value >1 is unsupported.
            conv_xy_first (bool): If True, spatial convolution comes before temporal conv

        BN related configs:
            norm (callable): a callable that constructs normalization layer, options
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer, options
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): 2plus1d conv layer.
    """
    if inner_channels is None:
        inner_channels = out_channels

    assert (
        groups == 1
    ), "Support for groups is not implemented in R2+1 convolution layer"
    assert (
        max(dilation) == 1 and min(dilation) == 1
    ), "Support for dillaiton is not implemented in R2+1 convolution layer"

    conv_t_module = nn.Conv3d(
        in_channels=in_channels if not conv_xy_first else inner_channels,
        out_channels=inner_channels if not conv_xy_first else out_channels,
        kernel_size=(kernel_size[0], 1, 1),
        stride=(stride[0], 1, 1),
        padding=(padding[0], 0, 0),
        bias=bias,
    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=inner_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()
    conv_xy_module = nn.Conv3d(
        in_channels=inner_channels if not conv_xy_first else in_channels,
        out_channels=out_channels if not conv_xy_first else inner_channels,
        kernel_size=(1, kernel_size[1], kernel_size[2]),
        stride=(1, stride[1], stride[2]),
        padding=(0, padding[1], padding[2]),
        bias=bias,
    )

    return Conv2plus1d(
        conv_t=conv_t_module,
        norm=norm_module,
        activation=activation_module,
        conv_xy=conv_xy_module,
        conv_xy_first=conv_xy_first,
    )


class Conv2plus1d(nn.Module):
    """
    Implementation of 2+1d Convolution by factorizing 3D Convolution into an 1D temporal
    Convolution and a 2D spatial Convolution with Normalization and Activation module
    in between:

                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)

    The 2+1d Convolution is used to build the R(2+1)D network.
    """

    def __init__(
        self,
        *,
        conv_t: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        conv_xy: nn.Module = None,
        conv_xy_first: bool = False,
    ) -> None:
        """
        Args:
            conv_t (torch.nn.modules): temporal convolution module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            conv_xy (torch.nn.modules): spatial convolution module.
            conv_xy_first (bool): If True, spatial convolution comes before temporal conv
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv_t is not None
        assert self.conv_xy is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
        x = self.norm(x) if self.norm else x
        x = self.activation(x) if self.activation else x
        x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
        return x

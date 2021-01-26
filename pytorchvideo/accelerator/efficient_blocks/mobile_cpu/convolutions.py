# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)

from .conv_helper import _Conv3dTemporalKernel3Decomposed, _Reshape


class Conv3dPwBnRelu(EfficientBlockBase):
    """
    Implements Conv3d + Bn + ReLu for pointwise layers.
    The conv layer has fixed kernel_size = (1,1,1),
    groups = 1, padding = 0, stride = 1, dilation = 1.

                          Input
                            |
                            ↓
                        conv3d (1x1x1)
                            ↓
                        BatchNorm (optional)
                            ↓
                        ReLU (optional)

    Conv3dPwBnRelu is in original form (for training) once instantiated. User can
    call convert() method to convert it into deployable form for deployment.

    convert_flag variable is to record whether the Conv3dPwBnRelu instance
    has been converted; Conv3dPwBnRelu is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Current implementation of this layer in QNNPACK is very efficient.
    Args:
        in_channels (int): number of input channels for conv3d 1x1x1.
        out_channels (int): number of output channels for conv3d 1x1x1.
        bias (bool): if true, use bias for conv.
        use_relu (bool): if true, use relu in block.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias=False,
        use_relu=True,
        use_bn=True,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        kernel = OrderedDict()
        kernel["conv"] = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            )
        if use_relu:
            kernel["relu"] = nn.ReLU(inplace=True)
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(
        self,
        input_blob_size: Tuple,
        **kwargs,
    ):
        """
        Converts Conv3d into equivalent Conv2d for Pytorch Mobile deployment.
        This conversion is done by first fuse conv3d with bn,
        convert conv3d into equivalent conv2d,
        and optionally fuse conv2d with relu.
        After conversion, the forwarding of this module becomes:
        Input (5d tensor) --> reshape (4d tensor) --> conv2d (4d tensor)
            --> reshape (5d tensor) --> output (5d tensor)
        Args:
            input_blob_size (tuple): blob size at the input of Conv3dPwBnRelu instance.
            kwargs (any): any extra keyword arguments from upstream unused by convert().
        """
        assert (
            self.convert_flag is False
        ), "Conv3dPwBnRelu: already converted, cannot be converted again"
        self.kernel.eval()
        # First fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])

        batch_size = input_blob_size[0]
        input_THW_tuple = input_blob_size[2:]
        self._input_tensor_reshape_size = (
            batch_size,
            self._in_channels,  # C
            input_THW_tuple[0] * input_THW_tuple[1],  # T*H
            input_THW_tuple[2],  # W
        )
        self._output_tensor_size = (
            batch_size,
            self._out_channels,  # C
            input_THW_tuple[0],  # T
            input_THW_tuple[1],  # H
            input_THW_tuple[2],  # W
        )
        conv2d_eq = nn.Conv2d(
            self._in_channels,
            self._out_channels,
            kernel_size=1,
            bias=(self.kernel.conv.bias is not None),
        )
        conv_state_dict = self.kernel.conv.state_dict()
        conv_state_dict["weight"] = conv_state_dict["weight"].squeeze(2)
        conv2d_eq.load_state_dict(conv_state_dict)
        self.kernel.conv = conv2d_eq
        # Fuse relu with conv after conv3d -> conv2d
        if hasattr(self.kernel, "relu"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "relu"])
        # Insert reshape layers before/after conv2d
        self.kernel = nn.Sequential(
            _Reshape(self._input_tensor_reshape_size),
            self.kernel,
            _Reshape(self._output_tensor_size),
        )
        self.convert_flag = True
        # Set new kernel in eval mode again
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d3x3x3DwBnRelu(EfficientBlockBase):
    """
    Implements Conv3d (3x3x3 dw) + (optional) Bn + (optional) ReLu for pointwise layers.
    The conv layer has fixed kernel_size = (3,3,3), depthwise, zero padding size of
    (1,1,1), temporal stride = 1, dilation = 1

                      Input
                        |
                        ↓
                    conv3d (3x3x3 dw)
                        ↓
                    BatchNorm (optional)
                        ↓
                    ReLU (optional)

    Current implementation of this layer in QNNPACK is reasonably efficient.

    convert_flag variable is to record whether the Conv3d3x3x3DwBnRelu instance
    has been converted; Conv3d3x3x3DwBnRelu is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of channels for conv3d 3x3x3 dw.
        spatial_stride (tuple length of 2): spatial stride for conv.
        bias (bool): if true, use bias for conv.
        use_relu (bool): if true, use relu in block.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    Current implementation of this layer in Pytorch Mobile is efficient.
    Sidenote: QNNPACK has best support for dw with 3x3 spatial kernel.
    For other spatial kernels like 7x7 dw, the efficiency may be lower.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_stride: int = 1,
        bias=False,
        use_relu=True,
        use_bn=True,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        kernel = OrderedDict()
        conv_stride = (1, spatial_stride, spatial_stride)
        kernel["conv"] = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 3, 3),
            stride=conv_stride,
            groups=in_channels,
            padding=1,
            bias=bias,
        )
        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                in_channels, eps=norm_eps, momentum=norm_momentum
            )
        if use_relu:
            kernel["relu"] = nn.ReLU(inplace=True)
        self.kernel = nn.Sequential(kernel)

        self.convert_flag = False

    def convert(
        self,
        input_blob_size: Tuple,
        **kwargs,
    ):
        """
        Converts Conv3d into equivalent Conv2d for efficient Pytorch Mobile deployment.
        Args:
            input_blob_size (tuple): blob size at the input of Conv3d3x3x3DwBnRelu
                instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert (
            self.convert_flag is False
        ), "Conv3d3x3x3DwBnRelu: already converted, cannot be converted twice."
        self.kernel.eval()
        # Fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])
        self.kernel.conv = _Conv3dTemporalKernel3Decomposed(
            self.kernel.conv, input_blob_size[2:]
        )
        """
        Since conv3d is converted into multiple conv2d,
        will not fuse conv with relu to keep arithmetic equivalency.
        """
        self.convert_flag = True
        # Set new kernel in eval mode again
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x

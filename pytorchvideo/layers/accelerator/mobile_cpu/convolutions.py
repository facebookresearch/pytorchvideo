# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)

from .activation_functions import supported_act_functions
from .conv_helper import (
    _Conv3dTemporalKernel1Decomposed,
    _Conv3dTemporalKernel3Decomposed,
    _Conv3dTemporalKernel5Decomposed,
    _Reshape,
)


class Conv3dPwBnAct(EfficientBlockBase):
    """
    Implements Conv3d + Bn + Activation for pointwise layers.
    The conv layer has fixed kernel_size = (1,1,1),
    groups = 1, padding = 0, stride = 1, dilation = 1.

                          Input
                            |
                            ↓
                        conv3d (1x1x1)
                            ↓
                        BatchNorm (optional)
                            ↓
                        Activation

    Conv3dPwBnAct is in original form (for training) once instantiated. User can
    call convert() method to convert it into deployable form for deployment.

    convert_flag variable is to record whether the Conv3dPwBnAct instance
    has been converted; Conv3dPwBnAct is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Current implementation of this layer in QNNPACK is very efficient.
    Args:
        in_channels (int): number of input channels for conv3d 1x1x1.
        out_channels (int): number of output channels for conv3d 1x1x1.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias=False,
        activation: str = "relu",
        use_bn=True,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.act = activation
        kernel = OrderedDict()
        kernel["conv"] = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            )
        assert (
            activation in supported_act_functions
        ), f"Conv3dPwBnAct: {activation} is not in supported_act_functions."
        kernel["act"] = supported_act_functions[activation]()
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
            input_blob_size (tuple): blob size at the input of Conv3dPwBnAct instance.
            kwargs (any): any extra keyword arguments from upstream unused by convert().
        """
        assert (
            self.convert_flag is False
        ), "Conv3dPwBnAct: already converted, cannot be converted again"
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
        # Convert activatiopn function
        self.kernel.act.convert(input_blob_size, **kwargs)
        # Fuse act with conv after conv3d -> conv2d if act is relu
        if self.act == "relu":
            self.kernel = torch.quantization.fuse_modules(
                self.kernel, ["conv", "act.act"]
            )
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


class Conv3d3x3x3DwBnAct(EfficientBlockBase):
    """
    Implements Conv3d (3x3x3 dw) + (optional) Bn + Activation layers.
    The conv layer has fixed kernel_size = (3,3,3), depthwise, zero padding size of
    (1,1,1), temporal stride = 1, dilation = 1

                      Input
                        |
                        ↓
                    conv3d (3x3x3 dw)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    Current implementation of this layer in QNNPACK is reasonably efficient.

    convert_flag variable is to record whether the Conv3d3x3x3DwBnAct instance
    has been converted; Conv3d3x3x3DwBnAct is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of channels for conv3d 3x3x3 dw.
        spatial_stride (tuple length of 2): spatial stride for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
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
        activation: str = "relu",
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
        assert (
            activation in supported_act_functions
        ), f"Conv3d3x3x3DwBnAct: {activation} is not in supported_act_functions."
        kernel["act"] = supported_act_functions[activation]()
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
            input_blob_size (tuple): blob size at the input of Conv3d3x3x3DwBnAct
                instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert (
            self.convert_flag is False
        ), "Conv3d3x3x3DwBnAct: already converted, cannot be converted twice."
        self.kernel.eval()
        # Fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])
        self.kernel.conv = _Conv3dTemporalKernel3Decomposed(
            self.kernel.conv, input_blob_size[2:]
        )
        # Convert activatiopn function
        self.kernel.act.convert(input_blob_size, **kwargs)
        """
        Since conv3d is converted into multiple conv2d,
        will not fuse conv with act to keep arithmetic equivalency.
        """
        self.convert_flag = True
        # Set new kernel in eval mode again
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3dTemporalKernel1BnAct(EfficientBlockBase):
    """
    Implements Conv3d + Bn + Activation where Conv3d has temporal kernel of 1.
    The conv layer has padding[0] = 0, stride[0] = 1, dilation[0] = 1.

                                  Input
                                    |
                                    ↓
                                conv3d (1xkxk)
                                    ↓
                                BatchNorm (optional)
                                    ↓
                                Activation

    Current implementation of this layer in QNNPACK is reasonably efficient
    (not as efficient as Conv3dPwBnAct for 1x1x1 kernel).
    Args:
        in_channels (int): number of input channels for conv3d 1x1x1.
        out_channels (int): number of output channels for conv3d 1x1x1.
        bias (bool): if true, use bias for conv.
        groups (int): number of groups for conv.
        spstial_kernel (int): spatial kernel for conv3d.
        spstial_stride (int): spatial stride for conv3d.
        spatial_padding (int): spatial padding for conv3d.
        spatial_dilation (int): spatial dilation for conv3d.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias=False,
        groups: int = 1,
        spatial_kernel: int = 1,
        spatial_stride: int = 1,
        spatial_padding: int = 0,
        spatial_dilation: int = 1,
        activation: str = "relu",
        use_bn=True,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
    ):
        super().__init__()

        kernel_size = (1, spatial_kernel, spatial_kernel)
        stride = (1, spatial_stride, spatial_stride)
        padding = (0, spatial_padding, spatial_padding)
        dilation = (1, spatial_dilation, spatial_dilation)
        kernel = OrderedDict()
        kernel["conv"] = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            )
        assert (
            activation in supported_act_functions
        ), f"Conv3dTemporalKernel1BnAct: {activation} is not in supported_act_functions."
        kernel["act"] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)

        self.convert_flag = False

    def convert(
        self,
        input_blob_size: Tuple,
        **kwargs,
    ):
        """
        Converts Conv3d into equivalent Conv2d for QNNPACK deployment.
        This conversion is done by first fuse conv3d with bn,
        convert conv3d into equivalent conv2d,
        and optionally fuse conv2d with relu.
        Args:
            input_blob_size (tuple): blob size at the input of
                Conv3dTemporalKernel1BnAct instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert (
            self.convert_flag is False
        ), "Conv3dTemporalKernel1BnAct: already converted, cannot be converted again"
        self.kernel.eval()
        # First fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])

        self.kernel.conv = _Conv3dTemporalKernel1Decomposed(
            self.kernel.conv, input_blob_size[2:]
        )
        # Convert activatiopn function
        self.kernel.act.convert(input_blob_size, **kwargs)

        self.convert_flag = True
        # Set new kernel in eval mode again
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d3x1x1BnAct(EfficientBlockBase):
    """
    Implements Conv3d (3x1x1) + (optional) Bn + Activation for pointwise layers.
    The conv layer has fixed kernel of (3, 1, 1), zero padding size of
    (1, 0, 0), stride = (1, 1, 1), dilation = 1.

                      Input
                        |
                        ↓
                    conv3d (3x1x1)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    For regular convolution (i.e., groups=1), current implementation of this layer in
    QNNPACK is reasonably efficient.
    For depthwise convolution (i.e., groups=out_channels), current implementation of this
    layer in QNNPACK is not efficient as Conv3d3x3x3DwBnRelu, as QNNPACK does not have
    optimization for 1x1 depthwise convolution. The latencies of fp32 operation are similar
    for Conv3d3x1x1BnAct and Conv3d3x3x3DwBnRelu, while with int8 operation Conv3d3x1x1BnAct
    is 1.5X slower than Conv3d3x3x3DwBnRelu.

    self.convert_flag property records whether the Conv3d3x1x1BnAct instance has been
    converted; Conv3d3x1x1BnAct is in original form if convert_flag is false, while it
    is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of input channels for conv3d 3x1x1.
        out_channels (int): number of output channels for conv3d 3x1x1.
        groups (int): number of groups for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        bias=False,
        activation: str = "relu",
        use_bn=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        super().__init__()
        kernel = OrderedDict()
        kernel["conv"] = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            groups=groups,
            padding=(1, 0, 0),
            bias=bias,
        )

        if groups == out_channels:
            logging.warn(
                (
                    "Conv3d3x1x1BnAct has low efficiency for depthwise conv. "
                    "Consider using Conv3d3x3x3DwBnRelu instead."
                )
            )

        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            )
        assert (
            activation in supported_act_functions
        ), f"Conv3d3x1x1BnAct: {activation} is not in supported_act_functions."
        kernel["act"] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(
        self,
        input_blob_size,
        **kwargs,
    ):
        """
        Converts Conv3d into equivalent Conv2d for Pytorch Mobile deployment

        """
        assert (
            self.convert_flag is False
        ), "Conv3d3x1x1BnAct: already converted, cannot be converted twice"
        self.kernel.eval()
        # Fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])
        self.kernel.conv = _Conv3dTemporalKernel3Decomposed(
            self.kernel.conv, input_blob_size[2:]
        )
        # Convert activation function
        self.kernel.act.convert(input_blob_size, **kwargs)
        # Since conv3d is converted into multiple conv2d, will not fuse conv with relu
        # to keep arithmetic equivalency.
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d5x1x1BnAct(EfficientBlockBase):
    """
    Implements Conv3d (5x1x1) + (optional) Bn + Activation for pointwise layers.
    The conv layer has fixed kernel of (5, 1, 1), zero padding size of
    (2, 0, 0), stride = (1, 1, 1), dilation = 1.

                      Input
                        |
                        ↓
                    conv3d (5x1x1)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    For regular convolution (i.e., groups=1), current implementation of this layer in
    QNNPACK is reasonably efficient.

    self.convert_flag property records whether the Conv3d5x1x1BnAct instance has been
    converted; Conv3d5x1x1BnAct is in original form if convert_flag is false, while it
    is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of input channels for conv3d 3x1x1.
        out_channels (int): number of output channels for conv3d 3x1x1.
        groups (int): number of groups for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        bias=False,
        activation: str = "relu",
        use_bn=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        super().__init__()
        kernel = OrderedDict()
        kernel["conv"] = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(5, 1, 1),
            groups=groups,
            padding=(2, 0, 0),
            bias=bias,
        )

        if use_bn:
            kernel["bn"] = nn.BatchNorm3d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            )
        assert (
            activation in supported_act_functions
        ), f"Conv3d5x1x1BnAct: {activation} is not in supported_act_functions."
        kernel["act"] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size, **kwargs):
        """
        Converts Conv3d into equivalent Conv2d for Pytorch Mobile deployment

        """
        assert (
            self.convert_flag is False
        ), "Conv3d5x1x1BnAct: already converted, cannot be converted twice"
        self.kernel.eval()
        # Fuse conv and bn if bn exists.
        if hasattr(self.kernel, "bn"):
            self.kernel = torch.quantization.fuse_modules(self.kernel, ["conv", "bn"])
        self.kernel.conv = _Conv3dTemporalKernel5Decomposed(
            self.kernel.conv, input_blob_size[2:]
        )
        # Convert activatiopn function
        self.kernel.act.convert(input_blob_size, **kwargs)
        # Since conv3d is converted into multiple conv2d, will not fuse conv with relu
        # to keep arithmetic equivalency.
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x

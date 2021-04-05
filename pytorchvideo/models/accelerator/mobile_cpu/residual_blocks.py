# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import OrderedDict
from typing import Optional, Tuple

import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)
from pytorchvideo.layers.accelerator.mobile_cpu.activation_functions import (
    supported_act_functions,
)
from pytorchvideo.layers.accelerator.mobile_cpu.attention import SqueezeExcitation
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d3x3x3DwBnAct,
    Conv3dPwBnAct,
    Conv3dTemporalKernel1BnAct,
)
from pytorchvideo.layers.utils import round_width


class X3dBottleneckBlock(EfficientBlockBase):
    """
    Implements a X3D style residual block with optional squeeze-excite (SE)
    using efficient blocks.

                    Input +----------------------+
                    |                            |
                    v                            |
                    conv3d[0] (1x1x1)            |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    activation[0]                |
                    |                            |
                    v                            |
                    conv3d[1] (3x3x3 dw)         |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    Squeeze-Excite (optional)    |
                    |                            |
                    v                            |
                    activation[1]                |
                    |                            |
                    v                            |
                    conv3d[2] (1x1x1)            |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    sum  <-----------------------+
                    |
                    v
                    activation[2]

    Args:
        in_channels (int): input channels for for 1x1x1 conv3d[0].
        mid_channels (int): channels for 3x3x3 dw conv3d[1].
        out_channels (int): output channels for 1x1x1 conv3d[2].
        spatial_stride (int): spatial stride for 3x3x3 dw conv3d[1].
        se_ratio (float): if > 0, apply SE to the 3x3x3 dw conv3d[1], with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        bias (tuple of bool): if bias[i] is true, use bias for conv3d[i].
        act_functions (tuple of str): act_functions[i] is the activation function after
            conv3d[i]. act_functions[i] should be a key in dict supported_act_functions
            (see activation_functions.py for more info about supported activations).
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (tuple of bool): if use_bn[i] is true, use batchnorm after conv3d[i].
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_residual: bool = True,
        spatial_stride: int = 1,
        se_ratio: float = 0.0625,
        act_functions: Optional[Tuple[str]] = ("relu", "relu", "relu"),
        bias: Optional[Tuple[bool]] = (False, False, False),
        use_bn: Optional[Tuple[bool]] = (True, True, True),
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
    ):
        super().__init__()

        # Residual projection
        self._use_residual = use_residual
        self._res_proj = None
        if self._use_residual:
            self._residual_add_func = nn.quantized.FloatFunctional()
            if (spatial_stride != 1) or (in_channels != out_channels):
                self._res_proj = Conv3dTemporalKernel1BnAct(
                    in_channels,
                    out_channels,
                    bias=False,
                    groups=1,
                    spatial_kernel=1,
                    spatial_stride=spatial_stride,
                    spatial_padding=0,
                    spatial_dilation=1,
                    activation="identity",
                    use_bn=True,
                )

        layers = OrderedDict()

        # 1x1x1 pointwise layer conv[0]
        assert (
            act_functions[0] in supported_act_functions
        ), f"{act_functions[0]} is not supported."
        layers["conv_0"] = Conv3dPwBnAct(
            in_channels,
            mid_channels,
            bias=bias[0],
            # If activation function is relu, just include that in convBnRelu block.
            activation=act_functions[0],
            use_bn=use_bn[0],
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
        )

        # 3x3x3 dw layer conv[1]
        self._spatial_stride = spatial_stride
        self._mid_channels = mid_channels
        assert (
            act_functions[1] in supported_act_functions
        ), f"{act_functions[1]} is not supported."
        layers["conv_1"] = Conv3d3x3x3DwBnAct(
            mid_channels,
            spatial_stride=self._spatial_stride,
            bias=bias[1],
            activation="identity",  # Will apply activation after SE.
            use_bn=use_bn[1],
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
        )
        if se_ratio > 0:
            layers["se"] = SqueezeExcitation(
                num_channels=mid_channels,
                num_channels_reduced=round_width(mid_channels, se_ratio),
                is_3d=True,
            )
        # Add activation function if act_functions[1].
        layers["act_func_1"] = supported_act_functions[act_functions[1]]()

        # Second 1x1x1 pointwise layer conv[2]
        self._out_channels = out_channels
        assert (
            act_functions[2] in supported_act_functions
        ), f"{act_functions[2]} is not supported."
        layers["conv_2"] = Conv3dPwBnAct(
            mid_channels,
            out_channels,
            bias=bias[2],
            # With residual, apply activation function externally after residual sum.
            activation="identity",
            use_bn=use_bn[2],
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
        )
        self.final_act = supported_act_functions[act_functions[2]]()

        self.layers = nn.Sequential(layers)

        self.convert_flag = False

    def forward(self, x):
        out = self.layers(x)
        if self._use_residual:
            if self._res_proj is not None:
                x = self._res_proj(x)
            out = self._residual_add_func.add(x, out)
        out = self.final_act(out)
        return out

    def convert(self, input_blob_size, *args, **kwargs):
        assert (
            self.convert_flag is False
        ), "X3dBottleneckBlock: already converted, cannot be converted twice"

        # Convert self.layers
        batch_size = input_blob_size[0]
        THW_size = tuple(input_blob_size[2:])
        if self._res_proj is not None:
            self._res_proj.convert(input_blob_size)
        self.layers.conv_0.convert(input_blob_size)
        # Update input_blob_size when necessary after each layer
        input_blob_size = (batch_size, self._mid_channels) + THW_size

        self.layers.conv_1.convert(input_blob_size)
        THW_size = (
            THW_size[0],
            THW_size[1] // self._spatial_stride,
            THW_size[2] // self._spatial_stride,
        )
        input_blob_size = (batch_size, self._mid_channels) + THW_size
        if hasattr(self.layers, "se"):
            self.layers.se.convert(input_blob_size)
        self.layers.act_func_1.convert(input_blob_size)
        self.layers.conv_2.convert(input_blob_size)
        input_blob_size = (batch_size, self._out_channels) + THW_size
        self.final_act.convert(input_blob_size)
        self.convert_flag = True

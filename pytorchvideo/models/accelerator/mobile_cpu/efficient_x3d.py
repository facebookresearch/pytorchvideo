# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import OrderedDict

import torch.nn as nn
from pytorchvideo.layers.accelerator.mobile_cpu.activation_functions import (
    supported_act_functions,
)
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d5x1x1BnAct,
    Conv3dPwBnAct,
    Conv3dTemporalKernel1BnAct,
)
from pytorchvideo.layers.accelerator.mobile_cpu.fully_connected import FullyConnected
from pytorchvideo.layers.accelerator.mobile_cpu.pool import AdaptiveAvgPool3dOutSize1

from .residual_blocks import X3dBottleneckBlock


class EfficientX3d(nn.Module):
    """
    This class implements an X3D network for classification with efficient blocks.
    Args:
        num_classes (int): Number of classes in classification.
        dropout (float): Dropout rate used for training the network.
        expansion (str): Expansion for X3D. Possible options: 'XS', 'S', 'M', 'L'.
        head_act (str): The activation function to be applied in head, should be a key
            in dict supported_act_functions (see activation_functions.py for more info
            about supported activations).
    """

    def __init__(
        self,
        num_classes: int = 400,
        dropout: float = 0.5,
        expansion: str = "XS",
        head_act: str = "identity",
    ):
        super().__init__()
        assert expansion in (
            "XS",
            "S",
            "M",
            "L",
        ), f"Expansion {expansion} not supported."
        # s1 - stem
        s1 = OrderedDict()
        s1["pathway0_stem_conv_xy"] = Conv3dTemporalKernel1BnAct(
            3,
            24,
            bias=False,
            groups=1,
            spatial_kernel=3,
            spatial_stride=2,
            spatial_padding=1,
            activation="identity",
            use_bn=False,
        )
        s1["pathway0_stem_conv"] = Conv3d5x1x1BnAct(
            24,
            24,
            bias=False,
            groups=24,
            use_bn=True,
        )
        self.s1 = nn.Sequential(s1)
        # s2 - res2
        s2 = OrderedDict()
        depth_s2 = 5 if expansion == "L" else 3
        for i_block in range(depth_s2):
            cur_block = X3dBottleneckBlock(
                in_channels=24,
                mid_channels=54,
                out_channels=24,
                use_residual=True,
                spatial_stride=(2 if i_block == 0 else 1),
                se_ratio=(0.0625 if (i_block % 2) == 0 else 0),
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
            s2[f"pathway0_res{i_block}"] = cur_block
        self.s2 = nn.Sequential(s2)
        # s3 - res3
        s3 = OrderedDict()
        depth_s3 = 10 if expansion == "L" else 5
        for i_block in range(depth_s3):
            cur_block = X3dBottleneckBlock(
                in_channels=(24 if i_block == 0 else 48),
                mid_channels=108,
                out_channels=48,
                use_residual=True,
                spatial_stride=(2 if i_block == 0 else 1),
                se_ratio=(0.0625 if (i_block % 2) == 0 else 0),
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
            s3[f"pathway0_res{i_block}"] = cur_block
        self.s3 = nn.Sequential(s3)
        # s4 - res4
        s4 = OrderedDict()
        depth_s4 = 25 if expansion == "L" else 11
        for i_block in range(depth_s4):
            cur_block = X3dBottleneckBlock(
                in_channels=(48 if i_block == 0 else 96),
                mid_channels=216,
                out_channels=96,
                use_residual=True,
                spatial_stride=(2 if i_block == 0 else 1),
                se_ratio=(0.0625 if (i_block % 2) == 0 else 0),
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
            s4[f"pathway0_res{i_block}"] = cur_block
        self.s4 = nn.Sequential(s4)
        # s5 - res5
        s5 = OrderedDict()
        depth_s5 = 15 if expansion == "L" else 7
        for i_block in range(depth_s5):
            cur_block = X3dBottleneckBlock(
                in_channels=(96 if i_block == 0 else 192),
                mid_channels=432,
                out_channels=192,
                use_residual=True,
                spatial_stride=(2 if i_block == 0 else 1),
                se_ratio=(0.0625 if (i_block % 2) == 0 else 0),
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
            s5[f"pathway0_res{i_block}"] = cur_block
        self.s5 = nn.Sequential(s5)
        # head
        head = OrderedDict()
        head["conv_5"] = Conv3dPwBnAct(
            in_channels=192,
            out_channels=432,
            bias=False,
            use_bn=True,
        )
        head["avg_pool"] = AdaptiveAvgPool3dOutSize1()
        head["lin_5"] = Conv3dPwBnAct(
            in_channels=432,
            out_channels=2048,
            bias=False,
            use_bn=False,
        )
        self.head = nn.Sequential(head)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.projection = FullyConnected(2048, num_classes, bias=True)
        assert head_act in supported_act_functions, f"{head_act} is not supported."
        self.act = supported_act_functions[head_act]()

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)

        return x


def create_x3d(
    *,
    # EfficientX3d model arguments.
    num_classes: int = 400,
    dropout: float = 0.5,
    expansion: str = "XS",
    head_act: str = "identity",
):
    """
    This function builds a X3D network with efficient blocks.
    Args:
        num_classes (int): Number of classes in classification.
        dropout (float): Dropout rate used for training the network.
        expansion (str): Expansion for X3D. Possible options: 'XS', 'S', 'M', 'L'.
        head_act (str): The activation function to be applied in head, should be a key
            in dict supported_act_functions (see activation_functions.py for more info
            about supported activations). Currently ReLU ('relu'), Swish ('swish'),
            Hardswish ('hswish'), Identity ('identity') are supported.
    """
    return EfficientX3d(
        num_classes=num_classes, dropout=dropout, expansion=expansion, head_act=head_act
    )

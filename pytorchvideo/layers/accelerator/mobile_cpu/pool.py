# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Tuple, Union

import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)
from pytorchvideo.accelerator.efficient_blocks.no_op_convert_block import (
    NoOpConvertBlock,
)


class AdaptiveAvgPool3dOutSize1(EfficientBlockBase):
    """
    Implements AdaptiveAvgPool3d with output (T, H, W) = (1, 1, 1). This operator has
    better efficiency than AdaptiveAvgPool for mobile CPU.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, **kwargs):
        """
        Converts AdaptiveAvgPool into AvgPool with constant kernel size for better
        efficiency.
        Args:
            input_blob_size (tuple): blob size at the input of
                AdaptiveAvgPool3dOutSize1 instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert (
            self.convert_flag is False
        ), "AdaptiveAvgPool3dOutSize1: already converted, cannot be converted again"
        kernel_size = input_blob_size[2:]
        self.pool = nn.AvgPool3d(kernel_size)
        self.convert_flag = True

    def forward(self, x):
        return self.pool(x)


class AdaptiveAvgPool2dOutSize1(EfficientBlockBase):
    """
    Implements AdaptiveAvgPool2d with output (H, W) = (1, 1). This operator has
    better efficiency than AdaptiveAvgPool for mobile CPU.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, **kwargs):
        """
        Converts AdaptiveAvgPool into AvgPool with constant kernel size for better
        efficiency.
        Args:
            input_blob_size (tuple): blob size at the input of
                AdaptiveAvgPool2dOutSize1 instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert (
            self.convert_flag is False
        ), "AdaptiveAvgPool2dOutSize1: already converted, cannot be converted again"
        kernel_size = input_blob_size[2:]
        self.pool = nn.AvgPool2d(kernel_size)
        self.convert_flag = True

    def forward(self, x):
        return self.pool(x)


class AdaptiveAvgPool3d(NoOpConvertBlock):
    """
    Implements AdaptiveAvgPool3d with any output (T, H, W) size. This operator is
    supported by QNNPACK for mobile CPU with resonable efficiency, and no change is
    made upon convert(). If the output (T, H, W) = (1, 1, 1), use AdaptiveAvgPool3dOutSize1
    for better efficiency.
    Args:
        output_size (int or tuple): when it is a tuple, the output (T, H, W) of pool
            will be equal to output_size. When it is an int, the output (T, H, W)
            will be equal to (output_size, output_size, output_size).
    """

    def __init__(
        self,
        output_size: Union[int, Tuple],
    ):
        super().__init__(model=nn.AdaptiveAvgPool3d(output_size))


class AdaptiveAvgPool2d(NoOpConvertBlock):
    """
    Implements AdaptiveAvgPool2d with any output (H, W) size. This operator is
    supported by QNNPACK for mobile CPU with resonable efficiency, and no change is
    made upon convert(). If the output (H, W) = (1, 1), use AdaptiveAvgPool2dOutSize1
    for better efficiency.
    Args:
        output_size (int or tuple): when it is a tuple, the output (H, W) of pool
            will be equal to output_size. When it is an int, the output (H, W)
            will be equal to (output_size, output_size).
    """

    def __init__(
        self,
        output_size: Union[int, Tuple],
    ):
        super().__init__(model=nn.AdaptiveAvgPool2d(output_size))

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.no_op_convert_block import (
    NoOpConvertBlock,
)


class FullyConnected(NoOpConvertBlock):
    """
    Implements fully connected layer. This operator is natively supported by QNNPACK for
    mobile CPU with good efficiency, and no change is made upon convert().
    Args:
        in_features (int): input channels for FC layer.
        out_features (int): output channels for FC layer.
        bias (bool): if True, bias is applied
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):

        super().__init__(model=nn.Linear(in_features, out_features, bias=bias))

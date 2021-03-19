# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn

from .efficient_block_base import EfficientBlockBase


class NoOpConvertBlock(EfficientBlockBase):
    """
    This class provides an interface with EfficientBlockBase for modules that do not
    need convert.
    Args:
        model (nn.Module): NoOpConvertBlock takes model as input and generate a wrapper
            instance of EfficientBlockBase with same functionality as model, with no change
            applied when convert() is called.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def convert(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self.model(x)

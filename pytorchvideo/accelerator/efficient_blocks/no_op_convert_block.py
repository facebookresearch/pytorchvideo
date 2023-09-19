# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn

from .efficient_block_base import EfficientBlockBase


class NoOpConvertBlock(EfficientBlockBase):
    """
    A class that provides an interface with EfficientBlockBase for modules that do not
    require conversion.

    Args:
        model (nn.Module): NoOpConvertBlock takes a model as input and generates a wrapper
            instance of EfficientBlockBase with the same functionality as the model. When
            `convert()` is called on this instance, no changes are applied.

    This class is designed for modules that do not need any conversion when integrated into
    an EfficientBlockBase. It takes an existing `model` and acts as a pass-through, forwarding
    input directly to the underlying model during the `forward` pass. When `convert()` is
    called, it simply does nothing, ensuring that no modifications are made to the model.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def convert(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self.model(x)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import abstractmethod

import torch.nn as nn


class EfficientBlockBase(nn.Module):
    """
    The EfficientBlockBase is the foundation for efficient blocks provided by PyTorchVideo's accelerator.
    These efficient blocks are designed for optimal efficiency on various target hardware devices.

    Each efficient block has two forms:
    - Original Form: This form is used during training. When an efficient block is instantiated,
        it is in its original form.
    - Deployable Form: This form is for deployment. Once the network is prepared for deployment,
        it can be converted into the deployable form to enable efficient execution on the target hardware.
        Conversion to the deployable form involves various optimizations such as operator fusion
        and kernel optimization.

    All efficient blocks must inherit from this base class and implement the following methods:
    - forward(): This method serves the same purpose as required by nn.Module.
    - convert(): Called to transform the block into its deployable form.

    Subclasses of EfficientBlockBase should provide implementations for these methods to tailor
    the behavior of the efficient block for specific use cases and target hardware.

    Note: This class is abstract, and its methods must be implemented in derived classes.
    """

    @abstractmethod
    def convert(self):
        """
        Abstract method to convert the efficient block into its deployable form.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Abstract method for the forward pass of the efficient block.
        """
        pass

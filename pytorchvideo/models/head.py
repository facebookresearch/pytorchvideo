# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes


def create_res_basic_head(
    *,
    # Projection configs.
    in_features: int,
    out_features: int,
    # Pooling configs.
    pool: Callable = nn.AvgPool3d,
    output_size: Tuple[int] = (1, 1, 1),
    pool_kernel_size: Tuple[int] = (1, 7, 7),
    pool_stride: Tuple[int] = (1, 1, 1),
    pool_padding: Tuple[int] = (0, 0, 0),
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = None,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:
    """
    Creates ResNet basic head. This layer performs an optional pooling operation
    followed by an optional dropout, a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.

    ::


                                        Pooling
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    Activation examples include: ReLU, Softmax, Sigmoid, and None.
    Pool3d examples include: AvgPool3d, MaxPool3d, AdaptiveAvgPool3d, and None.

    Args:

        in_features: input channel size of the resnet head.
        out_features: output channel size of the resnet head.

        pool (callable): a callable that constructs resnet head pooling layer,
            examples include: nn.AvgPool3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d, and
            None (not applying pooling).
        pool_kernel_size (tuple): pooling kernel size(s) when not using adaptive
            pooling.
        pool_stride (tuple): pooling stride size(s) when not using adaptive pooling.
        pool_padding (tuple): pooling padding size(s) when not using adaptive
            pooling.
        output_size (tuple): spatial temporal output size when using adaptive
            pooling.

        activation (callable): a callable that constructs resnet head activation
            layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
            applying activation).

        dropout_rate (float): dropout rate.

        output_with_global_average (bool): if True, perform global averaging on temporal
            and spatial dimensions and reshape output to batch_size x out_features.
    """
    if activation is None:
        activation_model = None
    elif activation == nn.Softmax:
        activation_model = activation(dim=1)
    else:
        activation_model = activation()

    if pool is None:
        pool_model = None
    elif pool == nn.AdaptiveAvgPool3d:
        pool_model = pool(output_size)
    else:
        pool_model = pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )

    if output_with_global_average:
        output_pool = nn.AdaptiveAvgPool3d(1)
    else:
        output_pool = None

    return ResNetBasicHead(
        proj=nn.Linear(in_features, out_features),
        activation=activation_model,
        pool=pool_model,
        dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
        output_pool=output_pool,
    )


class ResNetBasicHead(nn.Module):
    """
    ResNet basic head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_basic_head`.
    """

    def __init__(
        self,
        pool: nn.Module = None,
        dropout: nn.Module = None,
        proj: nn.Module = None,
        activation: nn.Module = None,
        output_pool: nn.Module = None,
    ) -> None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        x = x.permute((0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.permute((0, 4, 1, 2, 3))
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)

        if self.output_pool is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x

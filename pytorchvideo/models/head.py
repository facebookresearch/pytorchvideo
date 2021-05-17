# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from torchvision.ops import RoIAlign


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


def create_res_roi_pooling_head(
    *,
    # Projection configs.
    in_features: int,
    out_features: int,
    # RoI configs.
    resolution: Tuple,
    spatial_scale: float,
    sampling_ratio: int = 0,
    roi: Callable = RoIAlign,
    # Pooling configs.
    pool: Callable = nn.AvgPool3d,
    output_size: Tuple[int] = (1, 1, 1),
    pool_kernel_size: Tuple[int] = (1, 7, 7),
    pool_stride: Tuple[int] = (1, 1, 1),
    pool_padding: Tuple[int] = (0, 0, 0),
    pool_spatial: Callable = nn.MaxPool2d,
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = None,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:
    """
    Creates ResNet RoI head. This layer performs an optional pooling operation
    followed by an RoI projection, an optional 2D spatial pool, an optional dropout,
    a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.

                                        Pool3d
                                           ↓
                                       RoI Align
                                           ↓
                                        Pool2d
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
    RoI examples include: detectron2.layers.ROIAlign, detectron2.layers.ROIAlignRotated,
        tochvision.ops.RoIAlign and None
    Pool2d examples include: MaxPool2e, AvgPool2d, and None.

    Args:
        Projection related configs:
            in_features: input channel size of the resnet head.
            out_features: output channel size of the resnet head.

        RoI layer related configs:
            resolution (tuple): h, w sizes of the RoI interpolation.
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample interpolation. 0 to take samples densely.
            roi (callable): a callable that constructs the roi interpolation layer,
                examples include detectron2.layers.ROIAlign,
                detectron2.layers.ROIAlignRotated, and None.

        Pooling related configs:
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
            pool_spatial (callable): a callable that constructs the 2d pooling layer which
                follows the RoI layer, examples include: nn.AvgPool2d, nn.MaxPool2d, and
                None (not applying spatial pooling).

        Activation related configs:
            activation (callable): a callable that constructs resnet head activation
                layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
                applying activation).

        Dropout related configs:
            dropout_rate (float): dropout rate.

        Output related configs:
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

    return ResNetRoIHead(
        proj=nn.Linear(in_features, out_features),
        activation=activation_model,
        pool=pool_model,
        pool_spatial=pool_spatial(resolution, stride=1) if pool_spatial else None,
        roi_layer=roi(
            output_size=resolution,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
        ),
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


class ResNetRoIHead(nn.Module):
    """
    ResNet RoI head. This layer performs an optional pooling operation
    followed by an RoI projection, an optional 2D spatial pool, an optional dropout,
    a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.
                                        Pool3d
                                           ↓
                                       RoI Align
                                           ↓
                                        Pool2d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_roi_pooling_head`.
    """

    def __init__(
        self,
        pool: nn.Module = None,
        pool_spatial: nn.Module = None,
        roi_layer: nn.Module = None,
        dropout: nn.Module = None,
        proj: nn.Module = None,
        activation: nn.Module = None,
        output_pool: nn.Module = None,
    ) -> None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            pool_spatial (torch.nn.modules): pooling module.
            roi_spatial (torch.nn.modules): RoI (Ex: Align, pool) module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.tensor): input tensor
            bboxes (torch.tensor): Accociated bounding boxes.
                The format is N*5 (Index, X_1,Y_1,X_2,Y_2) if using RoIAlign
                and N*6 (Index, x_ctr, y_ctr, width, height, angle_degrees) if
                using RoIAlignRotated.
        """
        # Performs 3d pooling.
        if self.pool is not None:
            x = self.pool(x)
        # Performs roi layer using bboxes
        if self.roi_layer is not None:
            temporal_dim = x.shape[-3]
            if temporal_dim != 1:
                raise Exception(
                    "Temporal dimension should be 1. Consider modifying the pool layer."
                )
            x = torch.squeeze(x, -3)
            x = self.roi_layer(x, bboxes)
            # Performs spatial 2d pooling.
            if self.pool_spatial is not None:
                x = self.pool_spatial(x)
            x = x.unsqueeze(-3)
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

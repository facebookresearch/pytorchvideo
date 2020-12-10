# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple

import torch.nn as nn
from pytorchvideo.layers.convolutions import Conv2plus1d
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.resnet import BottleneckBlock, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem


def create_2plus1d_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = (1, 1, 1),
    conv_a_stride: Tuple[int] = (1, 1, 1),
    conv_a_padding: Tuple[int] = (0, 0, 0),
    conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    conv_b_stride: Tuple[int] = (2, 2, 2),
    conv_b_padding: Tuple[int] = (1, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    2plus1d bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).

    Args:
        Convolution related configs:
            dim_in (int): input channel size to the bottleneck block.
            dim_inner (int): intermediate channel size of the bottleneck.
            dim_out (int): output channel size of the bottleneck.
            conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            conv_a_stride (tuple): convolutional stride size(s) for conv_a.
            conv_a_padding (tuple): convolutional padding(s) for conv_a.
            conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            conv_b_stride (tuple): convolutional stride size(s) for conv_b.
            conv_b_padding (tuple): convolutional padding(s) for conv_b.
            conv_b_num_groups (int): number of groups for groupwise convolution for
                conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        Normalization related configs:
            norm (callable): a callable that constructs normalization layer, examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer, examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): 2plus1d bottleneck block.
    """
    # The first 1x1x1 Conv
    conv_a = nn.Conv3d(
        in_channels=dim_in,
        out_channels=dim_inner,
        kernel_size=conv_a_kernel_size,
        stride=conv_a_stride,
        padding=conv_a_padding,
        bias=False,
    )
    norm_a = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_a = None if activation is None else activation()

    # The 2+1d Conv
    conv_b = Conv2plus1d(
        conv_t=nn.Conv3d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            kernel_size=(conv_b_kernel_size[0], 1, 1),
            stride=(conv_b_stride[0], 1, 1),
            padding=(conv_b_padding[0], 0, 0),
            bias=False,
        ),
        norm=(
            None
            if norm is None
            else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
        ),
        activation=None if activation is None else activation(),
        conv_xy=nn.Conv3d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            kernel_size=(1, conv_b_kernel_size[1], conv_b_kernel_size[2]),
            stride=(1, conv_b_stride[1], conv_b_stride[2]),
            padding=(0, conv_b_padding[1], conv_b_padding[2]),
            bias=False,
        ),
    )
    norm_b = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b = None if activation is None else activation()

    # The second 1x1x1 Conv
    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return BottleneckBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=conv_b,
        norm_b=norm_b,
        act_b=act_b,
        conv_c=conv_c,
        norm_c=norm_c,
    )


def create_r2plus1d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.0,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (1, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_conv_b_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 2, 2),
    stage_bottleneck: Tuple[Callable] = (
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
    ),
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Softmax,
) -> nn.Module:
    """
    Build the R(2+1)D network from::
    A closer look at spatiotemporal convolutions for action recognition.
    Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri. CVPR 2018.

    R(2+1)D follows the ResNet style architecture including three parts: Stem,
    Stages and Head. The three parts are assembled in the following order:

                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head

    Args:
        Input clip configs:
            input_channel (int): number of channels for the input video clip.

        Model configs:
            model_depth (int): the depth of the resnet.
            model_num_class (int): the number of classes for the video dataset.
            dropout_rate (float): dropout rate.

        Normalization configs:
            norm (callable): a callable that constructs normalization layer.
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation configs:
            activation (callable): a callable that constructs activation layer.

        Stem configs:
            stem_dim_out (int): output channel size for stem.
            stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
            stem_conv_stride (tuple): convolutional stride size(s) of stem.

        Stage configs:
            stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
                for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
            stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
            stage_spatial_stride (tuple): the spatial stride for each stage.
            stage_temporal_stride (tuple): the temporal stride for each stage.
            stage_bottleneck (tuple): a callable that constructs bottleneck block layer
                for each stage. Examples include: create_bottleneck_block,
                create_2plus1d_bottleneck_block.

        Head configs:
            head_pool (callable): a callable that constructs resnet head pooling layer.
            head_pool_kernel_size (tuple): the pooling kernel size.
            head_output_size (tuple): the size of output tensor for head.
            head_activation (callable): a callable that constructs activation layer.

    Returns:
        (nn.Module): basic resnet.
    """
    # Number of blocks for different stages given the model depth.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    blocks = []
    # Create stem for R(2+1)D.
    stem = create_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=None,
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for R(2+1)D.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_b_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=stage_bottleneck[idx],
            conv_a_kernel_size=stage_conv_a_kernel_size[idx],
            conv_a_stride=[1, 1, 1],
            conv_a_padding=[size // 2 for size in stage_conv_a_kernel_size[idx]],
            conv_b_kernel_size=stage_conv_b_kernel_size[idx],
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[size // 2 for size in stage_conv_b_kernel_size[idx]],
            conv_b_num_groups=stage_conv_b_num_groups[idx],
            conv_b_dilation=stage_conv_b_dilation[idx],
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    # Create head for R(2+1)D.
    head = create_res_basic_head(
        in_features=stage_dim_in,
        out_features=model_num_class,
        pool=head_pool,
        output_size=head_output_size,
        pool_kernel_size=head_pool_kernel_size,
        dropout_rate=dropout_rate,
        activation=head_activation,
    )
    blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks))
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.stem import create_res_basic_stem


def create_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
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
    Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
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
        (nn.Module): resnet bottleneck block.
    """
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

    conv_b = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_b_kernel_size,
        stride=conv_b_stride,
        padding=conv_b_padding,
        bias=False,
        groups=conv_b_num_groups,
        dilation=conv_b_dilation,
    )
    norm_b = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b = None if activation is None else activation()

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


def create_acoustic_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    # Conv b f configs.
    conv_b_kernel_size: Tuple[int] = (1, 1, 1),
    conv_b_stride: Tuple[int] = (1, 1, 1),
    conv_b_padding: Tuple[int] = (0, 0, 0),
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
    Acoustic Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                           ---------------------------------
                           ↓                               ↓
                Temporal Conv3d (conv_b)        Spatial Conv3d (conv_b)
                           ↓                               ↓
                 Normalization (norm_b)         Normalization (norm_b)
                           ↓                               ↓
                   Activation (act_b)              Activation (act_b)
                           ↓                               ↓
                           ---------------------------------
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
        (nn.Module): resnet acoustic bottleneck block.
    """
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

    conv_b_1_kernel_size = [conv_b_kernel_size[0], 1, 1]
    conv_b_1_stride = [conv_b_stride[0], 1, 1]
    conv_b_1_padding = [conv_b_padding[0], 0, 0]

    conv_b_2_kernel_size = [1, conv_b_kernel_size[1], conv_b_kernel_size[2]]
    conv_b_2_stride = [1, conv_b_stride[1], conv_b_stride[2]]
    conv_b_2_padding = [0, conv_b_padding[1], conv_b_padding[2]]

    conv_b_1_num_groups, conv_b_2_num_groups = (conv_b_num_groups,) * 2
    conv_b_1_dilation = [conv_b_dilation[0], 1, 1]
    conv_b_2_dilation = [1, conv_b_dilation[1], conv_b_dilation[2]]

    conv_b_1 = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_b_1_kernel_size,
        stride=conv_b_1_stride,
        padding=conv_b_1_padding,
        bias=False,
        groups=conv_b_1_num_groups,
        dilation=conv_b_1_dilation,
    )
    norm_b_1 = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b_1 = None if activation is None else activation()

    conv_b_2 = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_b_2_kernel_size,
        stride=conv_b_2_stride,
        padding=conv_b_2_padding,
        bias=False,
        groups=conv_b_2_num_groups,
        dilation=conv_b_2_dilation,
    )
    norm_b_2 = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b_2 = None if activation is None else activation()

    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return SeparableBottleneckBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=nn.ModuleList([conv_b_2, conv_b_1]),
        norm_b=nn.ModuleList([norm_b_2, norm_b_1]),
        act_b=nn.ModuleList([act_b_2, act_b_1]),
        conv_c=conv_c,
        norm_c=norm_c,
    )


def create_res_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable,
    use_shortcut: bool = True,
    # Conv configs.
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
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
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).
    Transform examples include: BottleneckBlock.

    Args:
        Bottleneck block related configs:
            dim_in (int): input channel size to the bottleneck block.
            dim_inner (int): intermediate channel size of the bottleneck.
            dim_out (int): output channel size of the bottleneck.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_bottleneck_block.

        Convolution related configs:
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
            norm (callable): a callable that constructs normalization layer. Examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer. Examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): resnet basic block layer.
    """
    norm_model = None
    if norm is not None and dim_in != dim_out:
        norm_model = norm(num_features=dim_out)

    return ResBlock(
        branch1_conv=nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=tuple(map(np.prod, zip(conv_a_stride, conv_b_stride))),
            bias=False,
        )
        if dim_in != dim_out and use_shortcut
        else None,
        branch1_norm=norm_model if dim_in != dim_out and use_shortcut else None,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride,
            conv_a_padding=conv_a_padding,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride,
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation,
        ),
        activation=None if activation is None else activation(),
    )


def create_res_stage(
    *,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable,
    # Conv configs.
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
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
    Create Residual Stage, which composes sequential blocks that make up a ResNet. These
    blocks could be, for example, Residual blocks, Non-Local layers, or
    Squeeze-Excitation layers.


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).
    Bottleneck examples include: create_bottleneck_block.

    Args:
        Stage related configs:
            depth (init): number of blocks to create.
        Bottleneck block related configs:
            dim_in (int): input channel size to the bottleneck block.
            dim_inner (int): intermediate channel size of the bottleneck.
            dim_out (int): output channel size of the bottleneck.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_bottleneck_block.

        Convolution related configs:
            conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            conv_a_stride (tuple): convolutional stride size(s) for conv_a.
            conv_a_padding (tuple): convolutional padding(s) for conv_a.
            conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            conv_b_stride (tuple): convolutional stride size(s) for conv_b.
            conv_b_padding (tuple): convolutional padding(s) for conv_b.
            conv_b_num_groups (int): number of groups for groupwise convolution for
                conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        BN related configs:
            norm (callable): a callable that constructs normalization layer. Examples
                include nn.BatchNorm3d, and None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer. Examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): resnet basic stage layer.
    """
    res_blocks = []
    for ind in range(depth):
        block = create_res_block(
            dim_in=dim_in if ind == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride if ind == 0 else (1, 1, 1),
            conv_a_padding=conv_a_padding,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride if ind == 0 else (1, 1, 1),
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation,
        )
        res_blocks.append(block)
    return ResStage(res_blocks=nn.ModuleList(res_blocks))


def create_resnet(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = nn.MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
    ),
    stage_conv_b_kernel_size: Tuple[Tuple[int]] = (
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_spatial_stride: Tuple[int] = (2, 1, 1, 1),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_bottleneck_block,
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Softmax,
) -> nn.Module:
    """
    Build ResNet style models for video recognition. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:

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
            model_depth (int): the depth of the resnet. Options include: 50, 101, 152.
            model_num_class (int): the number of classes for the video dataset.
            dropout_rate (float): dropout rate.

        Normalization configs:
            norm (callable): a callable that constructs normalization layer.

        Activation configs:
            activation (callable): a callable that constructs activation layer.

        Stem configs:
            stem_dim_out (int): output channel size to stem.
            stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
            stem_conv_stride (tuple): convolutional stride size(s) of stem.
            stem_pool (callable): a callable that constructs resnet head pooling layer.
            stem_pool_kernel_size (tuple): pooling kernel size(s).
            stem_pool_stride (tuple): pooling stride size(s).

        Stage configs:
            stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
                for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
            stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
            stage_spatial_stride (tuple): the spatial stride for each stage.
            stage_temporal_stride (tuple): the temporal stride for each stage.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_bottleneck_block.

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
    blocks = []
    # Create stem for resnet.
    stem = create_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_a_stride = (stage_temporal_stride[idx], 1, 1)
        stage_conv_b_stride = (1, stage_spatial_stride[idx], stage_spatial_stride[idx])

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=stage_conv_a_kernel_size[idx],
            conv_a_stride=stage_conv_a_stride,
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


def create_acoustic_building_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = None,
    conv_a_stride: Tuple[int] = None,
    conv_a_padding: Tuple[int] = None,
    # Conv b f configs.
    conv_b_kernel_size: Tuple[int] = (1, 1, 1),
    conv_b_stride: Tuple[int] = (1, 1, 1),
    conv_b_padding: Tuple[int] = (0, 0, 0),
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
    Acoustic building block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                           ---------------------------------
                           ↓                               ↓
                Temporal Conv3d (conv_b)        Spatial Conv3d (conv_b)
                           ↓                               ↓
                 Normalization (norm_b)         Normalization (norm_b)
                           ↓                               ↓
                   Activation (act_b)              Activation (act_b)
                           ↓                               ↓
                           ---------------------------------
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
        (nn.Module): resnet acoustic bottleneck block.
    """
    # Conv b f configs.
    conv_b_1_kernel_size = [conv_b_kernel_size[0], 1, 1]
    conv_b_2_kernel_size = [1, conv_b_kernel_size[1], conv_b_kernel_size[2]]

    conv_b_1_stride = [conv_b_stride[0], 1, 1]
    conv_b_2_stride = [1, conv_b_stride[1], conv_b_stride[2]]

    conv_b_1_padding = [conv_b_padding[0], 0, 0]
    conv_b_2_padding = [0, conv_b_padding[1], conv_b_padding[2]]

    conv_b_1_num_groups, conv_b_2_num_groups = (conv_b_num_groups,) * 2

    conv_b_1_dilation = [conv_b_dilation[0], 1, 1]
    conv_b_2_dilation = [1, conv_b_dilation[1], conv_b_dilation[2]]

    conv_b_1 = nn.Conv3d(
        in_channels=dim_in,
        out_channels=dim_inner,
        kernel_size=conv_b_1_kernel_size,
        stride=conv_b_1_stride,
        padding=conv_b_1_padding,
        bias=False,
        groups=conv_b_1_num_groups,
        dilation=conv_b_1_dilation,
    )
    norm_b_1 = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b_1 = None if activation is None else activation()

    conv_b_2 = nn.Conv3d(
        in_channels=dim_in,
        out_channels=dim_inner,
        kernel_size=conv_b_2_kernel_size,
        stride=conv_b_2_stride,
        padding=conv_b_2_padding,
        bias=False,
        groups=conv_b_2_num_groups,
        dilation=conv_b_2_dilation,
    )
    norm_b_2 = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_b_2 = None if activation is None else activation()

    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )
    return SeparableBottleneckBlock(
        conv_a=None,
        norm_a=None,
        act_a=None,
        conv_b=nn.ModuleList([conv_b_1, conv_b_2]),
        norm_b=nn.ModuleList([norm_b_1, norm_b_2]),
        act_b=nn.ModuleList([act_b_1, act_b_2]),
        conv_c=conv_c,
        norm_c=norm_c,
    )


class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The builder can be found in `create_res_block`.
    """

    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch1_norm: nn.Module = None,
        branch2: nn.Module = None,
        activation: nn.Module = None,
    ) -> nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = x + self.branch2(x)
        else:
            residual = self.branch1_conv(x)
            if self.branch1_norm is not None:
                residual = self.branch1_norm(residual)
            x = residual + self.branch2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableBottleneckBlock(nn.Module):
    """
    Separable Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order. Requires a tuple of models to be
    provided to conv_b, norm_b, act_b to perform Convolution, Normalization, and
    Activations in parallel Separably.

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                 Conv3d(s) (conv_b), ...
                                         ↓ (↓)
                              Normalization(s) (norm_b), ...
                                         ↓ (↓)
                                 Activation(s) (act_b), ...
                                         ↓ (↓)
                                  Reduce (sum or cat)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)
    """

    def __init__(
        self,
        *,
        conv_a: nn.Module,
        norm_a: nn.Module,
        act_a: nn.Module,
        conv_b: nn.ModuleList,
        norm_b: nn.ModuleList,
        act_b: nn.ModuleList,
        conv_c: nn.Module,
        norm_c: nn.Module,
        reduce_method: str = "sum",
    ) -> None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules_list): convolutional module(s).
            norm_b (torch.nn.modules_list): normalization module(s).
            act_b (torch.nn.modules_list): activation module(s).
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
            reduce_method (str): if multiple conv_b is used, reduce the output with
                `sum`, or `cat`.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(
            op is not None for op in (self.conv_b, self.conv_c)
        ), f"{self.conv_a}, {self.conv_b}, {self.conv_c} has None"
        assert reduce_method in ["sum", "cat"]
        if self.norm_c is not None:
            # This flag is used for weight initialization.
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly forward every layer.
        # Branch2a, for example Tx1x1, BN, ReLU.
        if self.conv_a is not None:
            x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)

        # Branch2b, for example 1xHxW, BN, ReLU.
        output = []
        for ind in range(len(self.conv_b)):
            x_ = self.conv_b[ind](x)
            if self.norm_b[ind] is not None:
                x_ = self.norm_b[ind](x_)
            if self.act_b[ind] is not None:
                x_ = self.act_b[ind](x_)
            output.append(x_)
        if self.reduce_method == "sum":
            x = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduce_method == "cat":
            x = torch.cat(output, dim=1)

        # Branch2c, for example 1x1x1, BN.
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
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

    The builder can be found in `create_bottleneck_block`.
    """

    def __init__(
        self,
        *,
        conv_a: nn.Module = None,
        norm_a: nn.Module = None,
        act_a: nn.Module = None,
        conv_b: nn.Module = None,
        norm_b: nn.Module = None,
        act_b: nn.Module = None,
        conv_c: nn.Module = None,
        norm_c: nn.Module = None,
    ) -> None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules): convolutional module.
            norm_b (torch.nn.modules): normalization module.
            act_b (torch.nn.modules): activation module.
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b, self.conv_c))
        if self.norm_c is not None:
            # This flag is used for weight initialization.
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly forward every layer.
        # Branch2a, for example Tx1x1, BN, ReLU.
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)

        # Branch2b, for example 1xHxW, BN, ReLU.
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        if self.act_b is not None:
            x = self.act_b(x)

        # Branch2c, for example 1x1x1, BN.
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class ResStage(nn.Module):
    """
    ResStage composes sequential blocks that make up a ResNet. These blocks could be,
    for example, Residual blocks, Non-Local layers, or Squeeze-Excitation layers.

                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    The builder can be found in `create_res_stage`.
    """

    def __init__(self, res_blocks: nn.ModuleList) -> nn.Module:
        """
        Args:
            res_blocks (torch.nn.module_list): ResBlock module(s).
        """
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        return x

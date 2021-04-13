# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.stem import (
    create_acoustic_res_basic_stem,
    create_res_basic_stem,
)


def create_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    conv_a: Callable = nn.Conv3d,
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
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

    ::

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
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple): convolutional padding(s) for conv_a.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): resnet bottleneck block.
    """
    conv_a = conv_a(
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

    conv_b = conv_b(
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

    conv_c = conv_c(
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
    conv_a: Callable = nn.Conv3d,
    # Conv b f configs.
    conv_b_kernel_size: Tuple[int] = (1, 1, 1),
    conv_b_stride: Tuple[int] = (1, 1, 1),
    conv_b_padding: Tuple[int] = (0, 0, 0),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
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

    ::

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
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple): convolutional padding(s) for conv_a.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): resnet acoustic bottleneck block.
    """
    conv_a = conv_a(
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
    conv_b_1_stride = conv_b_stride
    conv_b_1_padding = [conv_b_padding[0], 0, 0]

    conv_b_2_kernel_size = [1, conv_b_kernel_size[1], conv_b_kernel_size[2]]
    conv_b_2_stride = conv_b_stride
    conv_b_2_padding = [0, conv_b_padding[1], conv_b_padding[2]]

    conv_b_1_num_groups, conv_b_2_num_groups = (conv_b_num_groups,) * 2
    conv_b_1_dilation = [conv_b_dilation[0], 1, 1]
    conv_b_2_dilation = [1, conv_b_dilation[1], conv_b_dilation[2]]

    conv_b_1 = conv_b(
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

    conv_b_2 = conv_b(
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

    conv_c = conv_c(
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
    use_shortcut: bool = False,
    branch_fusion: Callable = lambda x, y: x + y,
    # Conv configs.
    conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Tuple[int] = (1, 0, 0),
    conv_a: Callable = nn.Conv3d,
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
    conv_skip: Callable = nn.Conv3d,
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation_bottleneck: Callable = nn.ReLU,
    activation_block: Callable = nn.ReLU,
) -> nn.Module:
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


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
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        use_shortcut (bool): If true, use conv and norm layers in skip connection.
        branch_fusion (callable): a callable that constructs summation layer.
            Examples include: lambda x, y: x + y, OctaveSum.

        conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple): convolutional padding(s) for conv_a.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_skip (callable): a callable that constructs the conv_skip conv layer,
        examples include nn.Conv3d, OctaveConv, etc

        norm (callable): a callable that constructs normalization layer. Examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation_bottleneck (callable): a callable that constructs activation layer in
            bottleneck. Examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None
            (not performing activation).
        activation_block (callable): a callable that constructs activation layer used
            at the end of the block. Examples include: nn.ReLU, nn.Softmax, nn.Sigmoid,
            and None (not performing activation).

    Returns:
        (nn.Module): resnet basic block layer.
    """
    branch1_conv_stride = tuple(map(np.prod, zip(conv_a_stride, conv_b_stride)))
    norm_model = None
    if use_shortcut or (
        norm is not None and (dim_in != dim_out or np.prod(branch1_conv_stride) != 1)
    ):
        norm_model = norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)

    return ResBlock(
        branch1_conv=conv_skip(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=branch1_conv_stride,
            bias=False,
        )
        if (dim_in != dim_out or np.prod(branch1_conv_stride) != 1) or use_shortcut
        else None,
        branch1_norm=norm_model,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride,
            conv_a_padding=conv_a_padding,
            conv_a=conv_a,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride,
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            conv_b=conv_b,
            conv_c=conv_c,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation_bottleneck,
        ),
        activation=None if activation_block is None else activation_block(),
        branch_fusion=branch_fusion,
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
    conv_a_kernel_size: Union[Tuple[int], List[Tuple[int]]] = (3, 1, 1),
    conv_a_stride: Tuple[int] = (2, 1, 1),
    conv_a_padding: Union[Tuple[int], List[Tuple[int]]] = (1, 0, 0),
    conv_a: Callable = nn.Conv3d,
    conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    conv_b_stride: Tuple[int] = (1, 2, 2),
    conv_b_padding: Tuple[int] = (0, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
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

    ::


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
        depth (init): number of blocks to create.

        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.

        conv_a_kernel_size (tuple or list of tuple): convolutional kernel size(s)
            for conv_a. If conv_a_kernel_size is a tuple, use it for all blocks in
            the stage. If conv_a_kernel_size is a list of tuple, the kernel sizes
            will be repeated until having same length of depth in the stage. For
            example, for conv_a_kernel_size = [(3, 1, 1), (1, 1, 1)], the kernel
            size for the first 6 blocks would be [(3, 1, 1), (1, 1, 1), (3, 1, 1),
            (1, 1, 1), (3, 1, 1)].
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple or list of tuple): convolutional padding(s) for
            conv_a. If conv_a_padding is a tuple, use it for all blocks in
            the stage. If conv_a_padding is a list of tuple, the padding sizes
            will be repeated until having same length of depth in the stage.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc

        norm (callable): a callable that constructs normalization layer. Examples
            include nn.BatchNorm3d, and None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer. Examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): resnet basic stage layer.
    """
    res_blocks = []
    if isinstance(conv_a_kernel_size[0], int):
        conv_a_kernel_size = [conv_a_kernel_size]
    if isinstance(conv_a_padding[0], int):
        conv_a_padding = [conv_a_padding]
    # Repeat conv_a kernels until having same length of depth in the stage.
    conv_a_kernel_size = (conv_a_kernel_size * depth)[:depth]
    conv_a_padding = (conv_a_padding * depth)[:depth]

    for ind in range(depth):
        block = create_res_block(
            dim_in=dim_in if ind == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=conv_a_kernel_size[ind],
            conv_a_stride=conv_a_stride if ind == 0 else (1, 1, 1),
            conv_a_padding=conv_a_padding[ind],
            conv_a=conv_a,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride if ind == 0 else (1, 1, 1),
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            conv_b=conv_b,
            conv_c=conv_c,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation_bottleneck=activation,
            activation_block=activation,
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
    stage1_pool: Callable = None,
    stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
    stage_conv_a_kernel_size: Tuple[Union[Tuple[int], List[Tuple[int]]]] = (
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
    stage_spatial_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_bottleneck_block,
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Build ResNet style models for video recognition. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:

    ::

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

        input_channel (int): number of channels for the input video clip.

        model_depth (int): the depth of the resnet. Options include: 50, 101, 152.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.


        norm (callable): a callable that constructs normalization layer.

        activation (callable): a callable that constructs activation layer.

        stem_dim_out (int): output channel size to stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stem_pool (callable): a callable that constructs resnet head pooling layer.
        stem_pool_kernel_size (tuple): pooling kernel size(s).
        stem_pool_stride (tuple): pooling stride size(s).

        stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        stage_spatial_stride (tuple): the spatial stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck (callable): a callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.

        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_pool_kernel_size (tuple): the pooling kernel size.
        head_output_size (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.

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

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_a_kernel = stage_conv_a_kernel_size[idx]
        stage_conv_a_stride = (stage_temporal_stride[idx], 1, 1)
        stage_conv_a_padding = (
            [size // 2 for size in stage_conv_a_kernel]
            if isinstance(stage_conv_a_kernel[0], int)
            else [[size // 2 for size in sizes] for sizes in stage_conv_a_kernel]
        )

        stage_conv_b_stride = (1, stage_spatial_stride[idx], stage_spatial_stride[idx])

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=stage_conv_a_kernel,
            conv_a_stride=stage_conv_a_stride,
            conv_a_padding=stage_conv_a_padding,
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

        if idx == 0 and stage1_pool is not None:
            blocks.append(
                stage1_pool(
                    kernel_size=stage1_pool_kernel_size,
                    stride=stage1_pool_kernel_size,
                    padding=(0, 0, 0),
                )
            )

    head = create_res_basic_head(
        in_features=stage_dim_in,
        out_features=model_num_class,
        pool=head_pool,
        output_size=head_output_size,
        pool_kernel_size=head_pool_kernel_size,
        dropout_rate=dropout_rate,
        activation=head_activation,
        output_with_global_average=head_output_with_global_average,
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
    conv_a: Callable = None,
    # Conv b f configs.
    conv_b_kernel_size: Tuple[int] = (1, 1, 1),
    conv_b_stride: Tuple[int] = (1, 1, 1),
    conv_b_padding: Tuple[int] = (0, 0, 0),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = nn.Conv3d,
    conv_c: Callable = nn.Conv3d,
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

    ::


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

        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        conv_a_stride (tuple): convolutional stride size(s) for conv_a.
        conv_a_padding (tuple): convolutional padding(s) for conv_a.
        conv_a (callable): a callable that constructs the conv_a conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_b_stride (tuple): convolutional stride size(s) for conv_b.
        conv_b_padding (tuple): convolutional padding(s) for conv_b.
        conv_b_num_groups (int): number of groups for groupwise convolution for
            conv_b.
        conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        conv_b (callable): a callable that constructs the conv_b conv layer, examples
            include nn.Conv3d, OctaveConv, etc
        conv_c (callable): a callable that constructs the conv_c conv layer, examples
            include nn.Conv3d, OctaveConv, etc

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

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

    conv_b_1 = conv_b(
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

    conv_b_2 = conv_b(
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

    conv_c = conv_c(
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


def create_acoustic_resnet(
    *,
    # Model configs.
    input_channel: int = 2,
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (9, 9, 9),
    stem_conv_stride: Tuple[int] = (1, 1, 1),
    stem_conv_padding: Tuple[int] = (4, 4, 4),
    stem_pool: Callable = nn.MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    stem: Callable = create_acoustic_res_basic_stem,
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[int] = (3, 1, 1),
    stage_conv_a_padding: Tuple[int] = (1, 0, 0),
    stage_conv_b_kernel_size: Tuple[int] = (1, 3, 3),
    stage_conv_b_padding: Tuple[int] = (0, 1, 1),
    stage_conv_b_num_groups: int = 1,
    stage_conv_b_dilation: Tuple[int] = (1, 1, 1),
    stage_spatial_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 2, 2, 2),
    bottleneck: Tuple[Callable] = (
        create_acoustic_bottleneck_block,
        create_acoustic_bottleneck_block,
        create_bottleneck_block,
        create_bottleneck_block,
    ),
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Softmax,
    head_pool_kernel_size: Tuple[int] = (1, 1, 1),
) -> nn.Module:
    """
    Build ResNet style models for acoustic recognition. ResNet has three parts:
    Stem, Stages and Head. The three parts are assembled in the following order:

    ::

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

        input_channel (int): number of channels for the input video clip.
        input_clip_length (int): length of the input video clip.
        input_crop_size (int): spatial resolution of the input video clip.

        model_depth (int): the depth of the resnet.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.

        norm (callable): a callable that constructs normalization layer.

        activation (callable): a callable that constructs activation layer.

        stem_dim_out (int): output channel size to stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.
        stem_pool (callable): a callable that constructs resnet head pooling layer.
        stem_pool_kernel_size (tuple): pooling kernel size(s).
        stem_pool_stride (tuple): pooling stride size(s).
        stem (callable): a callable that constructs stem layer.
            Examples include: create_res_video_stem.

        stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups (int): number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation (tuple): dilation for 3D convolution for conv_b.
        stage_spatial_stride (tuple): the spatial stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck (callable): a callable that constructs bottleneck block
            layer.
            Examples include: create_bottleneck_block.

        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_output_size (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.

    Returns:
        (nn.Module): acoustic resnet that takes audio inputs in log-mel-spectrogram of
            shape B x 1 x 1 x T x F.
    """
    # Given a model depth, get the number of blocks for each stage.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
    assert model_depth in _MODEL_STAGE_DEPTH.keys()
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]
    assert len(bottleneck) == len(stage_depths)

    blocks = []
    # Create stem for resnet.
    stem = stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=stem_conv_padding,
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)
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
            bottleneck=bottleneck[idx],
            conv_a_kernel_size=stage_conv_a_kernel_size,
            conv_a_stride=stage_conv_a_stride,
            conv_a_padding=stage_conv_a_padding,
            conv_b_kernel_size=stage_conv_b_kernel_size,
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=stage_conv_b_padding,
            conv_b_num_groups=stage_conv_b_num_groups,
            conv_b_dilation=stage_conv_b_dilation,
            norm=norm,
            activation=activation,
        )
        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    # Create head for resnet.
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


class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


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
        branch_fusion: Callable = None,
    ) -> nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
            branch_fusion: (Callable): A callable or layer that combines branch1
                and branch2.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            x = self.branch_fusion(shortcut, self.branch2(x))
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableBottleneckBlock(nn.Module):
    """
    Separable Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order. Requires a tuple of models to be
    provided to conv_b, norm_b, act_b to perform Convolution, Normalization, and
    Activations in parallel Separably.

    ::


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

    ::


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

    ::


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

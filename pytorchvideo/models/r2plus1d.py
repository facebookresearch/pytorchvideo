# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from functools import partial
from typing import Callable, Tuple

import torch.nn as nn
from pytorchvideo.layers.convolutions import create_conv_2plus1d
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
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
    conv_a: Callable = nn.Conv3d,
    conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    conv_b_stride: Tuple[int] = (2, 2, 2),
    conv_b_padding: Tuple[int] = (1, 1, 1),
    conv_b_num_groups: int = 1,
    conv_b_dilation: Tuple[int] = (1, 1, 1),
    conv_b: Callable = create_conv_2plus1d,
    conv_c: Callable = nn.Conv3d,
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

    ::

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                  Conv(2+1)d (conv_b)
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
        (nn.Module): 2plus1d bottleneck block.
    """
    return create_bottleneck_block(
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
        conv_b=partial(
            create_conv_2plus1d,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation,
        ),
        conv_c=conv_c,
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
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
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Build the R(2+1)D network from::
    A closer look at spatiotemporal convolutions for action recognition.
    Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri. CVPR 2018.

    R(2+1)D follows the ResNet style architecture including three parts: Stem,
    Stages and Head. The three parts are assembled in the following order:

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

        model_depth (int): the depth of the resnet.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.

        norm (callable): a callable that constructs normalization layer.
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer.

        stem_dim_out (int): output channel size for stem.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.

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
        output_with_global_average=head_output_with_global_average,
    )
    blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks))

from typing import Callable, Tuple

import numpy as np
import torch.nn as nn
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.resnet import (
    ResNet,
    create_default_bottleneck_block,
    create_default_res_stage,
)
from pytorchvideo.models.stem import create_default_res_basic_stem


def create_default_csn(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 4,
    input_crop_size: int = 112,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[int] = (1, 1, 1),
    stage_conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    stage_conv_b_width_per_group: int = 1,
    stage_spatial_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 2, 2, 1),
    bottleneck: Callable = create_default_bottleneck_block,
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = nn.Softmax,
) -> nn.Module:
    """
    Build Channel-Separated Convolutional Networks (CSN):
    Video classification with channel-separated convolutional networks.
    Du Tran, Heng Wang, Lorenzo Torresani, Matt Feiszli. ICCV 2019.

    CSN follows the ResNet style architecture including three parts: Stem,
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

    Unlike the default ResNet, CSN uses depthwise convolution. To further
    reduce the computational cost, it uses low resolution (112x112), short
    clips (4 frames), different striding and kernel size, etc.

    Args:
        Input clip configs:
            input_channel (int): number of channels for the input video clip.
            input_clip_length (int): length of the input video clip.
            input_crop_size (int): spatial resolution of the input video clip.

        Model configs:
            model_depth (int): the depth of the resnet.
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

        Stage configs:
            stage_conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            stage_conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            stage_conv_b_width_per_group(int): the width of each group for conv_b. Set
                it to 1 for depthwise convolution.
            stage_spatial_stride (tuple): the spatial stride for each stage.
            stage_temporal_stride (tuple): the temporal stride for each stage.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_default_bottleneck_block.

        Head configs:
            head_pool (callable): a callable that constructs resnet head pooling layer.
            head_output_size (tuple): the size of output tensor for head.
            head_activation (callable): a callable that constructs activation layer.

    Returns:
        (nn.Module): the csn model.
    """
    # Number of blocks for different stages given the model depth.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
    # Create stem for CSN.
    stem = create_default_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=None,
        norm=norm,
        activation=activation,
    )

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    stages = []
    # Create each stage for CSN.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 8
        depth = stage_depths[idx]

        stage_conv_b_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_default_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=stage_conv_a_kernel_size,
            conv_a_stride=(1, 1, 1),
            conv_a_padding=[size // 2 for size in stage_conv_a_kernel_size],
            conv_b_kernel_size=stage_conv_b_kernel_size,
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[size // 2 for size in stage_conv_b_kernel_size],
            conv_b_num_groups=(stage_dim_inner // stage_conv_b_width_per_group),
            conv_b_dilation=(1, 1, 1),
            norm=norm,
            activation=activation,
        )

        stages.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    # Create head for CSN.
    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    head_pool_kernel_size = (
        input_clip_length // total_temporal_stride,
        input_crop_size // total_spatial_stride,
        input_crop_size // total_spatial_stride,
    )

    head = create_res_basic_head(
        in_features=stage_dim_in,
        out_features=model_num_class,
        pool=head_pool,
        output_size=head_output_size,
        pool_kernel_size=head_pool_kernel_size,
        dropout_rate=dropout_rate,
        activation=head_activation,
    )
    return ResNet(stem=stem, stages=stages, head=head)

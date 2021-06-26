# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.resnet import (
    create_acoustic_bottleneck_block,
    create_bottleneck_block,
)
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.stem import (
    create_acoustic_res_basic_stem,
    create_res_basic_stem,
)


# Note we expect audio data as (Time, 1, Frequency)
def create_audio_visual_slowfast(
    *,
    # SlowFast configs.
    slowfast_channel_reduction_ratio: Tuple[int] = (8, 2),
    slowfast_conv_channel_fusion_ratio: int = 2,
    fusion_builder: Callable[
        [int, int], nn.Module
    ] = None,  # Args: fusion_dim_in, stage_idx
    # Input clip configs.
    input_channels: Tuple[int] = (3, 3, 1),
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_outs: Tuple[int] = (64, 8, 32),
    stem_conv_kernel_sizes: Tuple[Tuple[int]] = ((1, 7, 7), (5, 7, 7), (9, 1, 9)),
    stem_conv_strides: Tuple[Tuple[int]] = ((1, 2, 2), (1, 2, 2), (1, 1, 1)),
    stem_pool: Tuple[Callable] = (nn.MaxPool3d, nn.MaxPool3d, None),
    stem_pool_kernel_sizes: Tuple[Tuple[int]] = ((1, 3, 3), (1, 3, 3), (1, 3, 3)),
    stem_pool_strides: Tuple[Tuple[int]] = ((1, 2, 2), (1, 2, 2), (1, 1, 1)),
    # Stage configs.
    stage_conv_a_kernel_sizes: Tuple[Tuple[Tuple[int]]] = (
        ((1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)),
        ((3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ),
    stage_conv_b_kernel_sizes: Tuple[Tuple[Tuple[int]]] = (
        ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
        ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
        ((3, 1, 3), (3, 1, 3), (3, 1, 3), (3, 1, 3)),
    ),
    stage_conv_b_num_groups: Tuple[Tuple[int]] = (
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
    ),
    stage_conv_b_dilations: Tuple[Tuple[Tuple[int]]] = (
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ),
    stage_spatial_strides: Tuple[Tuple[int]] = (
        (1, 2, 2, 2),
        (1, 2, 2, 2),
        (1, 2, 2, 2),
    ),
    stage_temporal_strides: Tuple[Tuple[int]] = (
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 2, 2, 2),
    ),
    bottleneck: Tuple[Tuple[Callable]] = (
        (
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
        ),
        (
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
        ),
        (
            create_acoustic_bottleneck_block,
            create_acoustic_bottleneck_block,
            create_bottleneck_block,
            create_bottleneck_block,
        ),
    ),
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_sizes: Tuple[Tuple[int]] = ((8, 7, 7), (32, 7, 7), (16, 1, 10)),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Model builder for AVSlowFast network.
    Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik, Christoph Feichtenhofer.
    "Audiovisual SlowFast Networks for Video Recognition."
    https://arxiv.org/abs/2001.08740

                             Slow Input  Fast Input  Audio Input
                                  ↓           ↓            ↓
                                 Stem       Stem         Stem
                                  ↓ ⭠ Fusion- ↓ ⭠ Fusion- ↓
                               Stage 1     Stage 1      Stage 1
                                  ↓ ⭠ Fusion- ↓ ⭠ Fusion- ↓
                                  .            .           .
                                  ↓            ↓           ↓
                               Stage N      Stage N     Stage N
                                  ↓ ⭠ Fusion- ↓ ⭠ Fusion- ↓
                                         ↓
                                       Head

    Args:
        SlowFast configs:
            slowfast_channel_reduction_ratio (int): Corresponds to the inverse of the channel
                reduction ratio, $\beta$F between the Slow and Fast pathways.
            slowfast_audio_reduction_ratio (int): Corresponds to the inverse of the channel
                reduction ratio, $\beta$A between the Slow and Audio pathways.
            slowfast_conv_channel_fusion_ratio (int): Ratio of channel dimensions
                between the Slow and Fast pathways.
            fusion_builder (Callable[[int, int], nn.Module]): Builder function for generating
                the fusion modules based on stage dimension and index

        Input clip configs:
            input_channels (tuple): number of channels for the input video clip.

        Model configs:
            model_depth (int): the depth of the resnet.
            model_num_class (int): the number of classes for the video dataset.
            dropout_rate (float): dropout rate.

        Normalization configs:
            norm (callable): a callable that constructs normalization layer.

        Activation configs:
            activation (callable): a callable that constructs activation layer.

        Stem configs:
            stem_function (Tuple[Callable]): a callable that constructs stem layer.
                Examples include create_res_basic_stem. Indexed by pathway
            stem_dim_outs (tuple): output channel size to stem.
            stem_conv_kernel_sizes (tuple): convolutional kernel size(s) of stem.
            stem_conv_strides (tuple): convolutional stride size(s) of stem.
            stem_pool (Tuple[Callable]): a callable that constructs resnet head pooling layer.
                Indexed by pathway
            stem_pool_kernel_sizes (tuple): pooling kernel size(s).
            stem_pool_strides (tuple): pooling stride size(s).

        Stage configs:
            stage_conv_a_kernel_sizes (tuple): convolutional kernel size(s) for conv_a.
            stage_conv_b_kernel_sizes (tuple): convolutional kernel size(s) for conv_b.
            stage_conv_b_num_groups (tuple): number of groups for groupwise convolution
                for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
            stage_conv_b_dilations (tuple): dilation for 3D convolution for conv_b.
            stage_spatial_strides (tuple): the spatial stride for each stage.
            stage_temporal_strides (tuple): the temporal stride for each stage.
            bottleneck (Tuple[Tuple[Callable]]): a callable that constructs bottleneck
                block layer. Examples include: create_bottleneck_block.
                Indexed by pathway and stage index

        Head configs:
            head_pool (callable): a callable that constructs resnet head pooling layer.
            head_output_sizes (tuple): the size of output tensor for head.
            head_activation (callable): a callable that constructs activation layer.
            head_output_with_global_average (bool): if True, perform global averaging on
                the head output.
    Returns:
        (nn.Module): SlowFast model.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_audio_visual_slowfast")

    # Number of blocks for different stages given the model depth.
    # 3 pathways, first is slow, second is fast, third is audio
    if fusion_builder is None:
        fusion_builder = AudioToSlowFastFusionBuilder(
            slowfast_channel_reduction_ratio=slowfast_channel_reduction_ratio[0],
            slowfast_audio_reduction_ratio=slowfast_channel_reduction_ratio[1],
            conv_fusion_channel_ratio=slowfast_conv_channel_fusion_ratio,
            conv_kernel_size=(7, 1, 1),
            conv_kernel_size_a=(5, 1, 1),
            conv_stride=(4, 1, 1),
            conv_stride_a=((16, 1, 1), (16, 1, 1), (8, 1, 1), (4, 1, 1), (2, 1, 1)),
            norm=norm,
            activation=activation,
        ).create_module

    return create_slowfast(
        slowfast_channel_reduction_ratio=slowfast_channel_reduction_ratio,
        slowfast_conv_channel_fusion_ratio=slowfast_conv_channel_fusion_ratio,
        fusion_builder=fusion_builder,
        # Input clip configs.
        input_channels=input_channels,
        # Model configs.
        model_depth=model_depth,
        model_num_class=model_num_class,
        dropout_rate=dropout_rate,
        # Normalization configs.
        norm=norm,
        # Activation configs.
        activation=activation,
        # Stem configs.
        stem_function=(
            create_res_basic_stem,
            create_res_basic_stem,
            create_acoustic_res_basic_stem,
        ),
        stem_dim_outs=stem_dim_outs,
        stem_conv_kernel_sizes=stem_conv_kernel_sizes,
        stem_conv_strides=stem_conv_strides,
        stem_pool=stem_pool,
        stem_pool_kernel_sizes=stem_pool_kernel_sizes,
        stem_pool_strides=stem_pool_strides,
        # Stage configs.
        stage_conv_a_kernel_sizes=stage_conv_a_kernel_sizes,
        stage_conv_b_kernel_sizes=stage_conv_b_kernel_sizes,
        stage_conv_b_num_groups=stage_conv_b_num_groups,
        stage_conv_b_dilations=stage_conv_b_dilations,
        stage_spatial_strides=stage_spatial_strides,
        stage_temporal_strides=stage_temporal_strides,
        bottleneck=bottleneck,
        # Head configs.
        head_pool=head_pool,
        head_pool_kernel_sizes=head_pool_kernel_sizes,
        head_output_size=head_output_size,
        head_activation=head_activation,
        head_output_with_global_average=head_output_with_global_average,
    )


class AudioToSlowFastFusionBuilder:
    def __init__(
        self,
        slowfast_channel_reduction_ratio: int,
        slowfast_audio_reduction_ratio: int,
        conv_fusion_channel_ratio: float,
        conv_kernel_size: Tuple[int],
        conv_kernel_size_a: Tuple[int],
        conv_stride: Union[Tuple[int], Tuple[Tuple[int]]],
        conv_stride_a: Union[Tuple[int], Tuple[Tuple[int]]],
        conv_fusion_channel_interm_dim: Union[int, float] = 0.25,  # also, 64
        conv_num_a: int = 2,
        norm: Callable = nn.BatchNorm3d,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        activation: Callable = nn.ReLU,
        max_stage_idx: int = 3,
    ) -> None:
        """
        Given a list of two tensors from Slow pathway and Fast pathway, fusion information
        from the Fast pathway to the Slow on through a convolution followed by a
        concatenation, then return the fused list of tensors from Slow and Fast pathway in
        order.
        Args:
            slowfast_channel_reduction_ratio (int): Reduction ratio from the stage dimension.
                Used to compute conv_dim_in = fusion_dim_in // slowfast_channel_reduction_ratio
            slowfast_audio_reduction_ratio (int): Audio Reduction ratio from the stage dimension.
                Used to compute conv_dim_in_a = fusion_dim_in // slowfast_audio_reduction_ratio
            conv_fusion_channel_ratio (int): channel ratio for the convolution used to fuse
                from Fast pathway to Slow pathway.
            conv_kernel_size (int): kernel size of the convolution used to fuse from Fast
                pathway to Slow pathway.
            conv_kernel_size_a (int): kernel size of the convolution used to fuse from Audio
                pathway to FastSlow pathway.
            conv_stride (int): stride size of the convolution used to fuse from Fast pathway
                to Slow pathway. Optionally indexed by stage.
            conv_stride_a (int): stride size of the convolution used to fuse from Audio pathway
                to FastSlow pathway. Optionally indexed by stage.
            conv_fusion_channel_interm_dim (Union[int, float]): When conv_num_a > 1 this value
                controls the dimensions of the intermediate conv
            conv_num_a (int): Number of intermediate conv for audio channel
            norm (callable): a callable that constructs normalization layer, examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.
            activation (callable): a callable that constructs activation layer, examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).
            max_stage_idx (int): Returns identity module if we exceed this
        """
        set_attributes(self, locals())

    def create_module(self, fusion_dim_in: int, stage_idx: int) -> nn.Module:
        """
        Creates the module for the given stage
        Args:
            fusion_dim_in (int): input stage dimension
            stage_idx (int): which stage this is
        """
        if stage_idx > self.max_stage_idx:
            return nn.Identity()

        conv_stride = (
            self.conv_stride[stage_idx]
            if isinstance(self.conv_stride[0], Tuple)
            else self.conv_stride
        )
        conv_stride_a = (
            self.conv_stride_a[stage_idx]
            if isinstance(self.conv_stride_a[0], Tuple)
            else self.conv_stride_a
        )

        conv_dim_in = fusion_dim_in // self.slowfast_channel_reduction_ratio
        conv_dim_in_a = fusion_dim_in // self.slowfast_audio_reduction_ratio
        fastslow_module = []
        fastslow_module.append(
            nn.Conv3d(
                conv_dim_in,
                int(conv_dim_in * self.conv_fusion_channel_ratio),
                kernel_size=self.conv_kernel_size,
                stride=conv_stride,
                padding=[k_size // 2 for k_size in self.conv_kernel_size],
                bias=False,
            )
        )
        if self.norm is not None:
            fastslow_module.append(
                self.norm(
                    num_features=conv_dim_in * self.conv_fusion_channel_ratio,
                    eps=self.norm_eps,
                    momentum=self.norm_momentum,
                )
            )
        if self.activation is not None:
            fastslow_module.append(self.activation())

        if isinstance(self.conv_fusion_channel_interm_dim, int):
            afs_fusion_interm_dim = self.conv_fusion_channel_interm_dim
        else:
            afs_fusion_interm_dim = int(
                conv_dim_in_a * self.conv_fusion_channel_interm_dim
            )

        block_audio_to_fastslow = []
        cur_dim_in = conv_dim_in_a
        for idx in range(self.conv_num_a):
            if idx == self.conv_num_a - 1:
                cur_stride = conv_stride_a
                cur_dim_out = int(
                    conv_dim_in * self.conv_fusion_channel_ratio + fusion_dim_in
                )
            else:
                cur_stride = (1, 1, 1)
                cur_dim_out = afs_fusion_interm_dim

            block_audio_to_fastslow.append(
                nn.Conv3d(
                    cur_dim_in,
                    cur_dim_out,
                    kernel_size=self.conv_kernel_size_a,
                    stride=cur_stride,
                    padding=[k_size // 2 for k_size in self.conv_kernel_size_a],
                    bias=False,
                )
            )
            if self.norm is not None:
                block_audio_to_fastslow.append(
                    self.norm(
                        num_features=cur_dim_out,
                        eps=self.norm_eps,
                        momentum=self.norm_momentum,
                    )
                )
            if self.activation is not None:
                block_audio_to_fastslow.append(self.activation())
            cur_dim_in = cur_dim_out

        return FuseAudioToFastSlow(
            block_fast_to_slow=nn.Sequential(*fastslow_module),
            block_audio_to_fastslow=nn.Sequential(*block_audio_to_fastslow),
        )


class FuseAudioToFastSlow(nn.Module):
    """
    Given a list of two tensors from Slow pathway and Fast pathway, fusion information
    from the Fast pathway to the Slow on through a convolution followed by a
    concatenation, then return the fused list of tensors from Slow and Fast pathway in
    order.
    """

    def __init__(
        self,
        block_fast_to_slow: nn.Module,
        block_audio_to_fastslow: nn.Module,
    ) -> None:
        """
        Args:
            conv_fast_to_slow (nn.module): convolution to perform fusion.
            norm (nn.module): normalization module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        x_a = x[2]
        fuse = self.block_fast_to_slow(x_f)

        # Reduce frequency dim
        average_a = torch.mean(x_a, dim=-1, keepdim=True)
        fuse_a = self.block_audio_to_fastslow(average_a)

        x_s_fuse = torch.cat([x_s, fuse], 1)
        print(x_s_fuse.size())
        return [fuse_a + x_s_fuse, x_f, x_a]

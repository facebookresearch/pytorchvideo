# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import MultiPathWayWithFuse, Net
from pytorchvideo.models.resnet import create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem


def create_slowfast(
    *,
    # SlowFast configs.
    slowfast_channel_reduction_ratio: Union[Tuple[int], int] = (8,),
    slowfast_conv_channel_fusion_ratio: int = 2,
    slowfast_fusion_conv_kernel_size: Tuple[int] = (
        7,
        1,
        1,
    ),  # deprecated, use fusion_builder
    slowfast_fusion_conv_stride: Tuple[int] = (
        4,
        1,
        1,
    ),  # deprecated, use fusion_builder
    fusion_builder: Callable[
        [int, int], nn.Module
    ] = None,  # Args: fusion_dim_in, stage_idx
    # Input clip configs.
    input_channels: Tuple[int] = (3, 3),
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_function: Tuple[Callable] = (
        create_res_basic_stem,
        create_res_basic_stem,
    ),
    stem_dim_outs: Tuple[int] = (64, 8),
    stem_conv_kernel_sizes: Tuple[Tuple[int]] = ((1, 7, 7), (5, 7, 7)),
    stem_conv_strides: Tuple[Tuple[int]] = ((1, 2, 2), (1, 2, 2)),
    stem_pool: Union[Callable, Tuple[Callable]] = (nn.MaxPool3d, nn.MaxPool3d),
    stem_pool_kernel_sizes: Tuple[Tuple[int]] = ((1, 3, 3), (1, 3, 3)),
    stem_pool_strides: Tuple[Tuple[int]] = ((1, 2, 2), (1, 2, 2)),
    # Stage configs.
    stage_conv_a_kernel_sizes: Tuple[Tuple[Tuple[int]]] = (
        ((1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)),
        ((3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)),
    ),
    stage_conv_b_kernel_sizes: Tuple[Tuple[Tuple[int]]] = (
        ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
        ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
    ),
    stage_conv_b_num_groups: Tuple[Tuple[int]] = ((1, 1, 1, 1), (1, 1, 1, 1)),
    stage_conv_b_dilations: Tuple[Tuple[Tuple[int]]] = (
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ),
    stage_spatial_strides: Tuple[Tuple[int]] = ((1, 2, 2, 2), (1, 2, 2, 2)),
    stage_temporal_strides: Tuple[Tuple[int]] = ((1, 1, 1, 1), (1, 1, 1, 1)),
    bottleneck: Union[Callable, Tuple[Tuple[Callable]]] = (
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
    ),
    # Head configs.
    head_pool: Callable = nn.AvgPool3d,
    head_pool_kernel_sizes: Tuple[Tuple[int]] = ((8, 7, 7), (32, 7, 7)),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> nn.Module:
    """
    Build SlowFast model for video recognition, SlowFast model involves a Slow pathway,
    operating at low frame rate, to capture spatial semantics, and a Fast pathway,
    operating at high frame rate, to capture motion at fine temporal resolution. The
    Fast pathway can be made very lightweight by reducing its channel capacity, yet can
    learn useful temporal information for video recognition. Details can be found from
    the paper:

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    ::

                             Slow Input  Fast Input
                                  ↓           ↓
                                 Stem       Stem
                                  ↓ ⭠ Fusion- ↓
                               Stage 1     Stage 1
                                  ↓ ⭠ Fusion- ↓
                                  .           .
                                  ↓           ↓
                               Stage N     Stage N
                                  ↓ ⭠ Fusion- ↓
                                         ↓
                                       Head

    Args:
        slowfast_channel_reduction_ratio (int): Corresponds to the inverse of the channel
            reduction ratio, $\beta$ between the Slow and Fast pathways.
        slowfast_conv_channel_fusion_ratio (int): Ratio of channel dimensions
            between the Slow and Fast pathways.
        DEPRECATED slowfast_fusion_conv_kernel_size (tuple): the convolutional kernel
            size used for fusion.
        DEPRECATED slowfast_fusion_conv_stride (tuple): the convolutional stride size
            used for fusion.
        fusion_builder (Callable[[int, int], nn.Module]): Builder function for generating
            the fusion modules based on stage dimension and index

        input_channels (tuple): number of channels for the input video clip.

        model_depth (int): the depth of the resnet.
        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.

        norm (callable): a callable that constructs normalization layer.

        activation (callable): a callable that constructs activation layer.

        stem_function (Tuple[Callable]): a callable that constructs stem layer.
            Examples include create_res_basic_stem. Indexed by pathway
        stem_dim_outs (tuple): output channel size to stem.
        stem_conv_kernel_sizes (tuple): convolutional kernel size(s) of stem.
        stem_conv_strides (tuple): convolutional stride size(s) of stem.
        stem_pool (Tuple[Callable]): a callable that constructs resnet head pooling layer.
            Indexed by pathway
        stem_pool_kernel_sizes (tuple): pooling kernel size(s).
        stem_pool_strides (tuple): pooling stride size(s).

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

        head_pool (callable): a callable that constructs resnet head pooling layer.
        head_output_sizes (tuple): the size of output tensor for head.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.
    Returns:
        (nn.Module): SlowFast model.
    """

    # Number of blocks for different stages given the model depth.
    _num_pathway = len(input_channels)
    _MODEL_STAGE_DEPTH = {
        18: (1, 1, 1, 1),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    # Fix up inputs
    if isinstance(slowfast_channel_reduction_ratio, int):
        slowfast_channel_reduction_ratio = (slowfast_channel_reduction_ratio,)
    if isinstance(stem_pool, Callable):
        stem_pool = (stem_pool,) * _num_pathway
    if isinstance(bottleneck, Callable):
        bottleneck = (bottleneck,) * len(stage_depths)
        bottleneck = (bottleneck,) * _num_pathway
    if fusion_builder is None:
        fusion_builder = FastToSlowFusionBuilder(
            slowfast_channel_reduction_ratio=slowfast_channel_reduction_ratio[0],
            conv_fusion_channel_ratio=slowfast_conv_channel_fusion_ratio,
            conv_kernel_size=slowfast_fusion_conv_kernel_size,
            conv_stride=slowfast_fusion_conv_stride,
            norm=norm,
            activation=activation,
            max_stage_idx=len(stage_depths) - 1,
        ).create_module

    # Build stem blocks.
    stems = []
    for pathway_idx in range(_num_pathway):
        stems.append(
            stem_function[pathway_idx](
                in_channels=input_channels[pathway_idx],
                out_channels=stem_dim_outs[pathway_idx],
                conv_kernel_size=stem_conv_kernel_sizes[pathway_idx],
                conv_stride=stem_conv_strides[pathway_idx],
                conv_padding=[
                    size // 2 for size in stem_conv_kernel_sizes[pathway_idx]
                ],
                pool=stem_pool[pathway_idx],
                pool_kernel_size=stem_pool_kernel_sizes[pathway_idx],
                pool_stride=stem_pool_strides[pathway_idx],
                pool_padding=[
                    size // 2 for size in stem_pool_kernel_sizes[pathway_idx]
                ],
                norm=norm,
                activation=activation,
            )
        )

    stages = []
    stages.append(
        MultiPathWayWithFuse(
            multipathway_blocks=nn.ModuleList(stems),
            multipathway_fusion=fusion_builder(
                fusion_dim_in=stem_dim_outs[0],
                stage_idx=0,
            ),
        )
    )

    # Build stages blocks.
    stage_dim_in = stem_dim_outs[0]
    stage_dim_out = stage_dim_in * 4
    for idx in range(len(stage_depths)):
        pathway_stage_dim_in = [
            stage_dim_in
            + stage_dim_in
            * slowfast_conv_channel_fusion_ratio
            // slowfast_channel_reduction_ratio[0],
        ]
        pathway_stage_dim_inner = [
            stage_dim_out // 4,
        ]
        pathway_stage_dim_out = [
            stage_dim_out,
        ]
        for reduction_ratio in slowfast_channel_reduction_ratio:
            pathway_stage_dim_in = pathway_stage_dim_in + [
                stage_dim_in // reduction_ratio
            ]
            pathway_stage_dim_inner = pathway_stage_dim_inner + [
                stage_dim_out // 4 // reduction_ratio
            ]
            pathway_stage_dim_out = pathway_stage_dim_out + [
                stage_dim_out // reduction_ratio
            ]

        stage = []
        for pathway_idx in range(_num_pathway):
            depth = stage_depths[idx]

            stage_conv_a_stride = (stage_temporal_strides[pathway_idx][idx], 1, 1)
            stage_conv_b_stride = (
                1,
                stage_spatial_strides[pathway_idx][idx],
                stage_spatial_strides[pathway_idx][idx],
            )
            stage.append(
                create_res_stage(
                    depth=depth,
                    dim_in=pathway_stage_dim_in[pathway_idx],
                    dim_inner=pathway_stage_dim_inner[pathway_idx],
                    dim_out=pathway_stage_dim_out[pathway_idx],
                    bottleneck=bottleneck[pathway_idx][idx],
                    conv_a_kernel_size=stage_conv_a_kernel_sizes[pathway_idx][idx],
                    conv_a_stride=stage_conv_a_stride,
                    conv_a_padding=[
                        size // 2
                        for size in stage_conv_a_kernel_sizes[pathway_idx][idx]
                    ],
                    conv_b_kernel_size=stage_conv_b_kernel_sizes[pathway_idx][idx],
                    conv_b_stride=stage_conv_b_stride,
                    conv_b_padding=[
                        size // 2
                        for size in stage_conv_b_kernel_sizes[pathway_idx][idx]
                    ],
                    conv_b_num_groups=stage_conv_b_num_groups[pathway_idx][idx],
                    conv_b_dilation=stage_conv_b_dilations[pathway_idx][idx],
                    norm=norm,
                    activation=activation,
                )
            )
        stages.append(
            MultiPathWayWithFuse(
                multipathway_blocks=nn.ModuleList(stage),
                multipathway_fusion=fusion_builder(
                    fusion_dim_in=stage_dim_out,
                    stage_idx=idx + 1,
                ),
            )
        )
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    if head_pool is None:
        pool_model = None
    elif head_pool == nn.AdaptiveAvgPool3d:
        pool_model = [head_pool(head_output_size[idx]) for idx in range(_num_pathway)]
    elif head_pool == nn.AvgPool3d:
        pool_model = [
            head_pool(
                kernel_size=head_pool_kernel_sizes[idx],
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            )
            for idx in range(_num_pathway)
        ]
    else:
        raise NotImplementedError(f"Unsupported pool_model type {pool_model}")

    stages.append(PoolConcatPathway(retain_list=False, pool=nn.ModuleList(pool_model)))
    head_in_features = stage_dim_in
    for reduction_ratio in slowfast_channel_reduction_ratio:
        head_in_features = head_in_features + stage_dim_in // reduction_ratio
    stages.append(
        create_res_basic_head(
            in_features=head_in_features,
            out_features=model_num_class,
            pool=None,
            output_size=head_output_size,
            dropout_rate=dropout_rate,
            activation=head_activation,
            output_with_global_average=head_output_with_global_average,
        )
    )
    return Net(blocks=nn.ModuleList(stages))


# TODO: move to pytorchvideo/layer once we have a common.py
class PoolConcatPathway(nn.Module):
    """
    Given a list of tensors, perform optional spatio-temporal pool and concatenate the
        tensors along the channel dimension.
    """

    def __init__(
        self,
        retain_list: bool = False,
        pool: Optional[nn.ModuleList] = None,
        dim: int = 1,
    ) -> None:
        """
        Args:
            retain_list (bool): if True, return the concatenated tensor in a list.
            pool (nn.module_list): if not None, list of pooling models for different
                pathway before performing concatenation.
            dim (int): dimension to performance concatenation.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.pool is not None:
            assert len(x) == len(self.pool)
        output = []
        for ind in range(len(x)):
            if x[ind] is not None:
                if self.pool is not None and self.pool[ind] is not None:
                    x[ind] = self.pool[ind](x[ind])
                output.append(x[ind])
        if self.retain_list:
            return [torch.cat(output, 1)]
        else:
            return torch.cat(output, 1)


class FastToSlowFusionBuilder:
    def __init__(
        self,
        slowfast_channel_reduction_ratio: int,
        conv_fusion_channel_ratio: float,
        conv_kernel_size: Tuple[int],
        conv_stride: Tuple[int],
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
            conv_fusion_channel_ratio (int): channel ratio for the convolution used to fuse
                from Fast pathway to Slow pathway.
            conv_kernel_size (int): kernel size of the convolution used to fuse from Fast
                pathway to Slow pathway.
            conv_stride (int): stride size of the convolution used to fuse from Fast pathway
                to Slow pathway.
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

        conv_dim_in = fusion_dim_in // self.slowfast_channel_reduction_ratio
        conv_fast_to_slow = nn.Conv3d(
            conv_dim_in,
            int(conv_dim_in * self.conv_fusion_channel_ratio),
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=[k_size // 2 for k_size in self.conv_kernel_size],
            bias=False,
        )
        norm_module = (
            None
            if self.norm is None
            else self.norm(
                num_features=conv_dim_in * self.conv_fusion_channel_ratio,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
            )
        )
        activation_module = None if self.activation is None else self.activation()
        return FuseFastToSlow(
            conv_fast_to_slow=conv_fast_to_slow,
            norm=norm_module,
            activation=activation_module,
        )


class FuseFastToSlow(nn.Module):
    """
    Given a list of two tensors from Slow pathway and Fast pathway, fusion information
    from the Fast pathway to the Slow on through a convolution followed by a
    concatenation, then return the fused list of tensors from Slow and Fast pathway in
    order.
    """

    def __init__(
        self,
        conv_fast_to_slow: nn.Module,
        norm: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
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
        fuse = self.conv_fast_to_slow(x_f)
        if self.norm is not None:
            fuse = self.norm(fuse)
        if self.activation is not None:
            fuse = self.activation(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

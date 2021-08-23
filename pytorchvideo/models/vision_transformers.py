# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from functools import partial
from typing import Tuple, Callable, List, Optional

import torch
import torch.nn as nn
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from torch.nn.common_types import _size_3_t

from .stem import create_conv_patch_embed


class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
    Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227

    ::

                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head


    The builder can be found in `create_mvit`.
    """

    def __init__(
        self,
        *,
        patch_embed: nn.Module,
        cls_positional_encoding: nn.Module,
        pos_drop: Optional[nn.Module],
        norm_patch_embed: Optional[nn.Module],
        blocks: nn.ModuleList,
        norm_embed: Optional[nn.Module],
        head: Optional[nn.Module],
    ) -> None:
        """
        Args:
            patch_embed (nn.Module): Patch embed module.
            cls_positional_encoding (nn.Module): Positional encoding module.
            pos_drop (Optional[nn.Module]): Dropout module after patch embed.
            blocks (nn.ModuleList): Stack of multi-scale transformer blocks.
            norm_layer (nn.Module): Normalization layer before head.
            head (Optional[nn.Module]): Head module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert hasattr(
            cls_positional_encoding, "patch_embed_shape"
        ), "cls_positional_encoding should have attribute patch_embed_shape."
        init_net_weights(self, init_std=0.02, style="vit")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)

        if self.pos_drop is not None:
            x = self.pos_drop(x)

        if self.norm_patch_embed is not None:
            x = self.norm_patch_embed(x)

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x


def create_multiscale_vision_transformers(
    *,
    spatial_size: int,
    temporal_size: int,
    cls_embed_on: bool = True,
    sep_pos_embed: bool = True,
    depth: int = 16,
    norm: str = "layernorm",
    # Patch embed config.
    input_channels: int = 3,
    patch_embed_dim: int = 96,
    conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
    conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
    conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
    enable_patch_embed_norm: bool = True,
    use_2d_patch: bool = False,
    # Attention block config.
    num_heads: int = 1,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    dropout_rate_block: float = 0.0,
    droppath_rate_block: float = 0.0,
    pooling_mode: str = "conv",
    pool_first: bool = False,
    embed_dim_mul: Optional[List[List[int]]] = None,
    atten_head_mul: Optional[List[List[int]]] = None,
    pool_q_stride_size: Optional[List[List[int]]] = None,
    pool_kv_stride_size: Optional[List[List[int]]] = None,
    pool_kv_stride_adaptive: Optional[_size_3_t] = None,
    pool_kvq_kernel: Optional[_size_3_t] = None,
    # Head config.
    head: Optional[Callable] = create_vit_basic_head,
    head_dropout_rate: float = 0.5,
    head_activation: Callable = None,
    head_num_classes: int = 400,
) -> nn.Module:
    """
    Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
    (ViT) is a specific case of MViT that only uses a single scale attention block.

    Args:
        spatial_size (int): Input video spatial resolution. It assumes the width and
            the height of the videos are the same.
        temporal_size (int): Number of frames in the input video.
        cls_embed_on (bool): If True, use cls embed in the model. Otherwise features
            are average pooled before going to the final classifier.
        sep_pos_embed (bool): If True, perform separate spatiotemporal embedding.
        depth (int): The depth of the model.
        norm (str): Normalization layer. It currently supports "layernorm".

        input_channels (int): Channel dimension of the input video.
        patch_embed_dim (int): Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel (Tuple[int]): Kernel size of the convolution for
            patchifing the video input.
        conv_patch_embed_stride (Tuple[int]): Stride size of the convolution for
            patchifing the video input.
        conv_patch_embed_padding (Tuple[int]): Padding size of the convolution for
            patchifing the video input.
        enable_patch_embed_norm (bool): If True, apply normalization after patchifing
            the video input.
        use_2d_patch (bool): If True, use 2D convolutions to get patch embed.
            Otherwise, use 3D convolutions.

        num_heads (int): Number of heads in the first transformer block.
        mlp_ratio (float): Mlp ratio which controls the feature dimension in the
            hidden layer of the Mlp block.
        qkv_bias (bool): If set to False, the qkv layer will not learn an additive
            bias. Default: False.
        dropout_rate_block (float): Dropout rate for the attention block.
        droppath_rate_block (float): Droppath rate for the attention block.
        pooling_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
            (average pooling), and "max" (max pooling).
        pool_first (bool): If set to True, pool is applied before qkv projection.
            Otherwise, pool is applied after qkv projection. Default: False.
        embed_dim_mul (Optional[List[List[int]]]): Dimension multiplication at layer i.
            If X is used, then the next block will increase the embed dimension by X
            times. Format: [depth_i, mul_dim_ratio].
        atten_head_mul (Optional[List[List[int]]]): Head dimension multiplication at
            layer i. If X is used, then the next block will increase the head by
            X times. Format: [depth_i, mul_dim_ratio].
        pool_q_stride_size (Optional[List[List[int]]]): List of stride sizes for the
            pool q at each layer. Format:
            [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_size (Optional[List[List[int]]]): List of stride sizes for the
            pool kv at each layer. Format:
            [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_adaptive (Optional[_size_3_t]): Initial kv stride size for the
            first block. The stride size will be further reduced at the layer where q
            is pooled with the ratio of the stride of q pooling. If
            pool_kv_stride_adaptive is set, then pool_kv_stride_size should be none.
        pool_kvq_kernel (Optional[_size_3_t]): Pooling kernel size for q and kv. It None,
            the kernel_size is [s + 1 if s > 1 else s for s in stride_size].

        head (Callable): Head model.
        head_dropout_rate (float): Dropout rate in the head.
        head_activation (Callable): Activation in the head.
        head_num_classes (int): Number of classes in the final classification head.

    Example usage (building a MViT_B model for Kinetics400):

        spatial_size = 224
        temporal_size = 16
        embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        pool_kv_stride_adaptive = [1, 8, 8]
        pool_kvq_kernel = [3, 3, 3]
        head_num_classes = 400
        MViT_B = create_multiscale_vision_transformers(
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            embed_dim_mul=embed_dim_mul,
            atten_head_mul=atten_head_mul,
            pool_q_stride_size=pool_q_stride_size,
            pool_kv_stride_adaptive=pool_kv_stride_adaptive,
            pool_kvq_kernel=pool_kvq_kernel,
            head_num_classes=head_num_classes,
        )
    """

    if use_2d_patch:
        assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
    if pool_kv_stride_adaptive is not None:
        assert (
            pool_kv_stride_size is None
        ), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
    if norm == "layernorm":
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    else:
        raise NotImplementedError("Only supports layernorm.")

    conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d

    patch_embed = create_conv_patch_embed(
        in_channels=input_channels,
        out_channels=patch_embed_dim,
        conv_kernel_size=conv_patch_embed_kernel,
        conv_stride=conv_patch_embed_stride,
        conv_padding=conv_patch_embed_padding,
        conv=conv_patch_op,
    )

    input_dims = [temporal_size, spatial_size, spatial_size]
    input_stirde = (
        (1,) + tuple(conv_patch_embed_stride)
        if use_2d_patch
        else conv_patch_embed_stride
    )

    patch_embed_dims = [
        input_dims[i] // input_stirde[i] for i in range(len(input_dims))
    ]
    cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_dims,
        sep_pos_embed=sep_pos_embed,
        has_cls=cls_embed_on,
    )

    dpr = [
        x.item() for x in torch.linspace(0, droppath_rate_block, depth)
    ]  # stochastic depth decay rule

    if dropout_rate_block > 0.0:
        pos_drop = nn.Dropout(p=dropout_rate_block)

    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

    norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None

    mvit_blocks = nn.ModuleList()

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    if pool_q_stride_size is not None:
        for i in range(len(pool_q_stride_size)):
            stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
                ]

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if pool_kv_stride_adaptive is not None:
        _stride_kv = pool_kv_stride_adaptive
        pool_kv_stride_size = []
        for i in range(depth):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            pool_kv_stride_size.append([i] + _stride_kv)

    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
                ]

    for i in range(depth):
        num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
        patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
        dim_out = round_width(
            patch_embed_dim,
            dim_mul[i + 1],
            divisor=round_width(num_heads, head_mul[i + 1]),
        )

        mvit_blocks.append(
            MultiScaleBlock(
                dim=patch_embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate_block,
                droppath_rate=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                pool_mode=pooling_mode,
                has_cls_embed=cls_embed_on,
                pool_first=pool_first,
            )
        )

    embed_dim = dim_out
    norm_embed = norm_layer(embed_dim)
    if head is not None:
        head_model = head(
            in_features=embed_dim,
            out_features=head_num_classes,
            seq_pool_type="cls" if cls_embed_on else "mean",
            dropout_rate=head_dropout_rate,
            activation=head_activation,
        )
    else:
        head_model = None

    return MultiscaleVisionTransformers(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
        norm_patch_embed=norm_patch_embed,
        blocks=mvit_blocks,
        norm_embed=norm_embed,
        head=head_model,
    )

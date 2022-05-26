# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.weight_init import init_net_weights
from torch.nn.common_types import _size_2_t, _size_3_t

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
        patch_embed: Optional[nn.Module],
        cls_positional_encoding: nn.Module,
        pos_drop: Optional[nn.Module],
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

        assert hasattr(
            cls_positional_encoding, "patch_embed_shape"
        ), "cls_positional_encoding should have method patch_embed_shape."

        self.patch_embed = patch_embed or torch.nn.Identity()
        self.cls_positional_encoding = cls_positional_encoding
        self.pos_drop = pos_drop or torch.nn.Identity()
        self.blocks = blocks
        self.norm_embed = norm_embed or torch.nn.Identity()
        self.head = head or torch.nn.Identity()

        init_net_weights(self, init_std=0.02, style="vit")

    def _get_bn_w_b(self, bn, repeat=1):
        w_bn = torch.diag(
            bn.weight.div(torch.sqrt(bn.eps + bn.running_var)).repeat(repeat)
        )

        b_bn = (
            bn.bias
            - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        ).repeat(repeat)
        return w_bn, b_bn

    def fuse_norm_before_linear(self, bn, linear):
        if bn is None:
            return linear
        w_bn, b_bn = self._get_bn_w_b(bn)
        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(linear.weight, w_bn)
        fused_linear.bias.data[:] = (
            torch.matmul(linear.weight, b_bn) + linear.bias
            if linear.bias is not None
            else torch.matmul(linear.weight, b_bn)
        )
        return fused_linear

    def fuse_norm_after_linear(self, linear, bn):
        if bn is None:
            return linear
        assert linear.in_features % bn.bias.shape[0] == 0
        num_heads = linear.in_features // bn.bias.shape[0]
        w_bn, b_bn = self._get_bn_w_b(bn, repeat=num_heads)

        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(w_bn, linear.weight)
        fused_linear.bias.data[:] = (
            torch.matmul(w_bn, linear.bias) + b_bn if linear.bias is not None else b_bn
        )
        return fused_linear

    def fuse_bn(self):
        assert not self.training
        for blk in self.blocks:
            # fuse self.norm1
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_before_linear(blk.norm1, blk.attn.q)
                blk.attn.k = self.fuse_norm_before_linear(blk.norm1, blk.attn.k)
                blk.attn.v = self.fuse_norm_before_linear(blk.norm1, blk.attn.v)
            else:
                blk.attn.qkv = self.fuse_norm_before_linear(blk.norm1, blk.attn.qkv)
            blk.norm1 = nn.Identity()

            # fuse the bn in attention
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_after_linear(blk.attn.q, blk.attn.norm_q)
                blk.attn.k = self.fuse_norm_after_linear(blk.attn.k, blk.attn.norm_k)
                blk.attn.v = self.fuse_norm_after_linear(blk.attn.v, blk.attn.norm_v)
            else:
                w_q, w_k, w_v = blk.attn.qkv.weight.chunk(3)
                b_q, b_k, b_v = blk.attn.qkv.bias.chunk(3)
                tmp_q = nn.Linear(w_q.shape[1], w_q.shape[0], bias=True)
                tmp_k = nn.Linear(w_k.shape[1], w_k.shape[0], bias=True)
                tmp_v = nn.Linear(w_v.shape[1], w_v.shape[0], bias=True)
                tmp_q.weight.data[:] = w_q
                tmp_k.weight.data[:] = w_k
                tmp_v.weight.data[:] = w_v
                tmp_q.bias.data[:] = b_q
                tmp_k.bias.data[:] = b_k
                tmp_v.bias.data[:] = b_v
                tmp_q = self.fuse_norm_after_linear(tmp_q, blk.attn.norm_q)
                tmp_k = self.fuse_norm_after_linear(tmp_k, blk.attn.norm_k)
                tmp_v = self.fuse_norm_after_linear(tmp_v, blk.attn.norm_v)
                blk.attn.qkv.weight.data[:] = torch.cat(
                    [tmp_q.weight.data, tmp_k.weight.data, tmp_v.weight.data], dim=0
                )
                blk.attn.qkv.bias.data[:] = torch.cat(
                    [tmp_q.bias.data, tmp_k.bias.data, tmp_v.bias.data], dim=0
                )

            blk.attn.norm_q = nn.Identity()
            blk.attn.norm_k = nn.Identity()
            blk.attn.norm_v = nn.Identity()

            # fuse self.norm2
            blk.mlp.fc1 = self.fuse_norm_before_linear(blk.norm2, blk.mlp.fc1)
            if blk.dim != blk.dim_out:
                blk.proj = self.fuse_norm_before_linear(blk.norm2, blk.proj)
            blk.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)
        x = self.pos_drop(x)

        thw = self.cls_positional_encoding.patch_embed_shape()
        for blk in self.blocks:
            x, thw = blk(x, thw)
        x = self.norm_embed(x)
        x = self.head(x)
        return x


def create_multiscale_vision_transformers(
    *,
    spatial_size: _size_2_t,
    temporal_size: int,
    cls_embed_on: bool = True,
    sep_pos_embed: bool = True,
    depth: int = 16,
    norm: str = "layernorm",
    # Patch embed config.
    enable_patch_embed: bool = True,
    input_channels: int = 3,
    patch_embed_dim: int = 96,
    conv_patch_embed_kernel: Tuple[int] = (3, 7, 7),
    conv_patch_embed_stride: Tuple[int] = (2, 4, 4),
    conv_patch_embed_padding: Tuple[int] = (1, 3, 3),
    enable_patch_embed_norm: bool = False,
    use_2d_patch: bool = False,
    # Attention block config.
    num_heads: int = 1,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    dropout_rate_block: float = 0.0,
    droppath_rate_block: float = 0.0,
    pooling_mode: str = "conv",
    pool_first: bool = False,
    residual_pool: bool = False,
    depthwise_conv: bool = True,
    bias_on: bool = True,
    separate_qkv: bool = True,
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
    # The default model definition is not TorchScript-friendly.
    # Set create_scriptable_model=True to create a TorchScriptable model.
    create_scriptable_model: bool = False,
    multiscale_vit_class: Callable = MultiscaleVisionTransformers,
) -> nn.Module:
    """
    Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
    (ViT) is a specific case of MViT that only uses a single scale attention block.

    Args:
        spatial_size (_size_2_t): Input video spatial resolution (H, W). If a single
            int is given, it assumes the width and the height are the same.
        temporal_size (int): Number of frames in the input video.
        cls_embed_on (bool): If True, use cls embed in the model. Otherwise features
            are average pooled before going to the final classifier.
        sep_pos_embed (bool): If True, perform separate spatiotemporal embedding.
        depth (int): The depth of the model.
        norm (str): Normalization layer. It currently supports "layernorm".

        enable_patch_embed (bool): If true, patchify the input video. If false, it
            assumes the input should have the feature dimension of patch_embed_dim.
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
            bias. Default: True.
        dropout_rate_block (float): Dropout rate for the attention block.
        droppath_rate_block (float): Droppath rate for the attention block.
        pooling_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
            (average pooling), and "max" (max pooling).
        pool_first (bool): If set to True, pool is applied before qkv projection.
            Otherwise, pool is applied after qkv projection. Default: False.
        residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
        depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
        bias_on (bool): Whether use biases for linear layers.
        separate_qkv (bool): Whether to use separate or one layer for qkv projections.
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
        multiscale_vit_class (Callable): MViT transformer class. Default to
            MultiscaleVisionTransformers.

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
        block_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        attn_norm_layer = partial(nn.LayerNorm, eps=1e-6)
    elif norm == "batchnorm":
        norm_layer = None
        block_norm_layer = nn.BatchNorm1d
        attn_norm_layer = nn.BatchNorm3d
    else:
        raise NotImplementedError("Only supports layernorm.")
    if create_scriptable_model:
        assert (
            norm == "batchnorm"
        ), "The scriptable model supports only the batchnorm-based model."
        warnings.warn(
            "`create_scriptable_model` is deprecated. MultiscaleVisionTransformers"
            " now supports scripting without this flag.",
            DeprecationWarning,
        )

    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d

    patch_embed = (
        create_conv_patch_embed(
            in_channels=input_channels,
            out_channels=patch_embed_dim,
            conv_kernel_size=conv_patch_embed_kernel,
            conv_stride=conv_patch_embed_stride,
            conv_padding=conv_patch_embed_padding,
            conv=conv_patch_op,
        )
        if enable_patch_embed
        else None
    )

    input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
    input_stirde = (
        (1,) + tuple(conv_patch_embed_stride)
        if use_2d_patch
        else conv_patch_embed_stride
    )

    patch_embed_shape = (
        [input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
        if enable_patch_embed
        else input_dims
    )

    cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_shape,
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
                norm_layer=block_norm_layer,
                attn_norm_layer=attn_norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                pool_mode=pooling_mode,
                has_cls_embed=cls_embed_on,
                pool_first=pool_first,
                residual_pool=residual_pool,
                bias_on=bias_on,
                depthwise_conv=depthwise_conv,
                separate_qkv=separate_qkv,
            )
        )

    embed_dim = dim_out
    norm_embed = None if norm_layer is None else norm_layer(embed_dim)
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

    return multiscale_vit_class(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
        blocks=mvit_blocks,
        norm_embed=norm_embed,
        head=head_model,
    )

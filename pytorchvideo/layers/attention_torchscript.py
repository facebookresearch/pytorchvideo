# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional, Tuple

import numpy
import torch
import torch.fx
import torch.nn as nn
from torch.nn.common_types import _size_3_t

from .drop_path import DropPath


class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.

    ::

                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        dropout_rate: float = 0.0,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_on)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_on)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@torch.fx.wrap
def _unsqueeze_dims_fx(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor, tensor_dim


@torch.jit.script
def _unsqueeze_dims_jit(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    return _unsqueeze_dims_fx(tensor)


@torch.fx.wrap
def _squeeze_dims_fx(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.squeeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor


@torch.jit.script
def _squeeze_dims_jit(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    return _squeeze_dims_fx(tensor, tensor_dim)


def _pre_attention_pool(
    tensor: torch.Tensor,
    thw_shape: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int, int, int]]:
    """
    Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


    Args:
        tensor (torch.Tensor): Input tensor.
        pool (Optional[Callable]): Pool operation that is applied to the input tensor.
            If pool is none, return the input tensor.
        thw_shape (List): The shape of the input tensor (before flattening).
        has_cls_embed (bool): Whether the input tensor contains cls token. Pool
            operation excludes cls token.
        norm: (Optional[Callable]): Optional normalization operation applied to
         tensor after pool.

    Returns:
        tensor (torch.Tensor): Input tensor after pool.
        thw_shape (List[int]): Output tensor shape (before flattening).
    """
    if torch.jit.is_scripting():
        tensor, tensor_dim = _unsqueeze_dims_jit(tensor)
    else:
        tensor, tensor_dim = _unsqueeze_dims_fx(tensor)
    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    return tensor, (B, N, L, C, T, H, W, tensor_dim)


def _post_attention_pool(
    tensor: torch.Tensor,
    thw_shape: List[int],
) -> Tuple[torch.Tensor, List[int]]:

    B, N, L, C, T, H, W, tensor_dim = thw_shape
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if torch.jit.is_scripting():
        tensor = _squeeze_dims_jit(tensor, tensor_dim)
    else:
        tensor = _squeeze_dims_fx(tensor, tensor_dim)

    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.

    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed: bool = True,
        pool_mode: str = "conv",
        pool_first: bool = False,
        residual_pool: bool = True,
        depthwise_conv: bool = True,
        bias_on: bool = True,
        separate_qkv: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """

        super().__init__()
        assert pool_mode in ["conv", "avg", "max"]
        assert not pool_first
        assert not has_cls_embed
        assert not separate_qkv
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True if bias_on else False)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            kernel_q is not None
            and numpy.prod(kernel_q) == 1
            and numpy.prod(stride_q) == 1
        ):
            kernel_q = None
        if (
            kernel_kv is not None
            and numpy.prod(kernel_kv) == 1
            and numpy.prod(stride_kv) == 1
        ):
            kernel_kv = None

        if pool_mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if pool_mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if kernel_q is not None
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if kernel_kv is not None
                else None
            )
        elif pool_mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_q is not None
                else None
            )
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
        else:
            raise NotImplementedError(f"Unsupported model {pool_mode}")

    def _qkv_proj(
        self,
        q: torch.Tensor,
        q_size: List[int],
        k: torch.Tensor,
        k_size: List[int],
        v: torch.Tensor,
        v_size: List[int],
        batch_size: List[int],
        chan_size: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = (
            self.q(q)
            .reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k)
            .reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v)
            .reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    def _qkv_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        thw_shape: List[int],
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        if self.pool_q is None:
            q_shape = thw_shape
        else:
            q, q_shape = _pre_attention_pool(
                q, [thw_shape[0], thw_shape[1], thw_shape[2]]
            )
            q = nn.functional.gelu(q)
            q = self.pool_q(q)
            q, q_shape = _post_attention_pool(
                q,
                q_shape,
            )

        if self.pool_k is None:
            k_shape = thw_shape
        else:
            k, k_shape = _pre_attention_pool(
                k,
                [thw_shape[0], thw_shape[1], thw_shape[2]],
            )
            k = nn.functional.gelu(k)
            k = self.pool_k(k)
            k, k_shape = _post_attention_pool(
                k,
                k_shape,
            )
        if self.pool_v is None:
            v_shape = thw_shape
        else:
            v, v_shape = _pre_attention_pool(
                v,
                [thw_shape[0], thw_shape[1], thw_shape[2]],
            )
            v = nn.functional.gelu(v)
            v = self.pool_v(v)
            v, v_shape = _post_attention_pool(
                v,
                v_shape,
            )
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int]:
        q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
        k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
        v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
        return q_N, k_N, v_N

    def _reshape_qkv_to_seq(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_N: int,
        v_N: int,
        k_N: int,
        B: int,
        C: int,
    ) -> Tuple[int]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        if self.residual_pool:
            x = (attn @ v + q).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        return x, q_shape


class ScriptableMultiScaleBlock(nn.Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        droppath_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_norm_layer: nn.Module = nn.LayerNorm,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        pool_mode: str = "conv",
        has_cls_embed: bool = True,
        pool_first: bool = False,
        residual_pool: bool = False,
        depthwise_conv: bool = True,
        bias_on: bool = True,
        separate_qkv: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        assert not pool_first
        assert not separate_qkv
        self.dim = dim
        self.dim_out = dim_out
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=attn_norm_layer,
            has_cls_embed=has_cls_embed,
            pool_mode=pool_mode,
            pool_first=pool_first,
            residual_pool=residual_pool,
            bias_on=bias_on,
            depthwise_conv=depthwise_conv,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
            bias_on=bias_on,
        )
        self.proj = (
            nn.Linear(dim, dim_out, bias=bias_on) if dim != dim_out else nn.Identity()
        )

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """

        x_block, thw_shape_new = self.attn(
            x,
            thw_shape,
        )

        if self.pool_skip is None:
            x_res = x
        else:
            x_res, res_shape = _pre_attention_pool(
                x, [thw_shape[0], thw_shape[1], thw_shape[2]]
            )
            x_res = self.pool_skip(x_res)
            x_res, _ = _post_attention_pool(x_res, res_shape)

        x = x_res + self.drop_path(x_block)
        x_norm = x
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new

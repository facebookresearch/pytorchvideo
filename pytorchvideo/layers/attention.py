# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, List, Optional, Tuple

import numpy
import torch

try:
    import torch.fx
except Exception as _:
    pass
import torch.nn as nn
from torch.nn.common_types import _size_3_t

from .drop_path import DropPath


@torch.fx.wrap
def _unsqueeze_dims_fx(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Unsqueezes dimensions of a 3D tensor to make it 4D if needed.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the modified tensor and its dimension.
        
    If the input tensor has 3 dimensions, it adds a new dimension at the second position to make
    it 4D. If the tensor already has 4 dimensions, it does nothing.

    Example:
    ```
    input_tensor = torch.randn(32, 3, 64, 64)
    modified_tensor, new_dim = _unsqueeze_dims_fx(input_tensor)
    ```
    """
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
    """
    JIT script version of _unsqueeze_dims_fx.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the modified tensor and its dimension.
    """
    return _unsqueeze_dims_fx(tensor)


@torch.fx.wrap
def _squeeze_dims_fx(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    """
    Squeezes dimensions of a 4D tensor to make it 3D if needed.

    Args:
        tensor (torch.Tensor): The input tensor.
        tensor_dim (int): The original dimension of the tensor.

    Returns:
        torch.Tensor: The modified tensor.
        
    If the input tensor has 4 dimensions and `tensor_dim` is 3, it removes the second dimension
    to make it 3D. If the tensor already has 3 dimensions and `tensor_dim` is 3, it does nothing.

    Example:
    ```
    input_tensor = torch.randn(32, 1, 64, 64)
    modified_tensor = _squeeze_dims_fx(input_tensor, 3)
    ```
    """
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.squeeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor


@torch.jit.script
def _squeeze_dims_jit(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    """
    JIT script version of _squeeze_dims_fx.

    Args:
        tensor (torch.Tensor): The input tensor.
        tensor_dim (int): The original dimension of the tensor.

    Returns:
        torch.Tensor: The modified tensor.
    """
    return _squeeze_dims_fx(tensor, tensor_dim)


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

    Args:
        in_features (int): Input feature dimension.
        hidden_features (Optional[int]): Hidden feature dimension (default is input dimension).
        out_features (Optional[int]): Output feature dimension (default is input dimension).
        act_layer (Callable): Activation layer applied after the first linear layer.
        dropout_rate (float): Dropout rate after each linear layer (0.0 by default).
        bias_on (bool): Whether to use biases for linear layers (True by default).

    Example:
    ```
    mlp_block = Mlp(in_features=256, hidden_features=512, dropout_rate=0.1)
    output = mlp_block(input_tensor)
    ```
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
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

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        return x


class _AttentionPool(torch.nn.Module):
    def __init__(
        self,
        pool: Optional[torch.nn.Module],
        has_cls_embed: bool,
        norm: Optional[torch.nn.Module],
    ) -> None:
        """Apply pool to a flattened input (given pool operation and the unflattened shape).


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
            pool (Optional[torch.nn.Module]): Pooling operation applied to the input tensor.
                If None, no pooling is applied.
            has_cls_embed (bool): Indicates whether the input tensor contains a cls token.
                The pooling operation excludes the cls token if present.
            norm (Optional[torch.nn.Module]): Optional normalization operation applied to
                the tensor after pooling.

        This class applies a specified pooling operation to a flattened input tensor, preserving the
        spatial structure. If the input tensor contains a cls token, the pooling operation excludes it.
        An optional normalization operation can be applied after pooling.

        Example:
        ```
        pool_layer = _AttentionPool(pool=torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
                                    has_cls_embed=True,
                                    norm=torch.nn.LayerNorm((16, 16, 16)))
        output_tensor, output_shape = pool_layer(input_tensor, [32, 16, 16])
        ```
        """
        super().__init__()
        self.has_pool = pool is not None
        self.pool = pool if pool is not None else torch.nn.Identity()

        self.has_cls_embed = has_cls_embed
        if norm is not None:
            self.norm_before_pool = isinstance(
                norm, (torch.nn.BatchNorm3d, torch.nn.Identity)
            )
            self.has_norm = True
            self.norm = norm
        else:
            self.norm_before_pool = False
            self.has_norm = False
            self.norm = torch.nn.Identity()

    def forward(
        self, tensor: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Applies the specified pooling operation to the input tensor while preserving spatial structure.

        Args:
            tensor (torch.Tensor): Input tensor.
            thw_shape (List[int]): The shape of the input tensor (before flattening).

        Returns:
            torch.Tensor: Output tensor after pooling.
            List[int]: Output tensor shape (before flattening).

        This method reshapes the input tensor, applies the pooling operation, and restores the
        original shape. If normalization is used, it can be applied before or after pooling
        based on the configuration.
        """
        if not self.has_pool:
            return tensor, thw_shape
        tensor_dim = tensor.ndim

        if torch.jit.is_scripting():
            tensor, tensor_dim = _unsqueeze_dims_jit(tensor)
        else:
            tensor, tensor_dim = _unsqueeze_dims_fx(tensor)

        cls_tok: torch.Tensor = torch.tensor(0)  # For typing/torchscriptability
        if self.has_cls_embed:
            cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

        B, N, L, C = tensor.shape
        T, H, W = thw_shape
        tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        if self.norm_before_pool:
            # If use BN, we apply norm before pooling instead of after pooling.
            tensor = self.norm(tensor)
            # We also empirically find that adding a GELU here is beneficial.
            tensor = torch.nn.functional.gelu(tensor)

        tensor = self.pool(tensor)

        thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
        L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
        tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
        if self.has_cls_embed:
            tensor = torch.cat((cls_tok, tensor), dim=2)
        if self.has_norm and not self.norm_before_pool:
            tensor = self.norm(tensor)

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
               [dim expand]     [dim expand]      [dim expand]
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

    _version = 3

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_q: _size_3_t = (1, 1, 1),
        kernel_kv: _size_3_t = (1, 1, 1),
        stride_q: _size_3_t = (1, 1, 1),
        stride_kv: _size_3_t = (1, 1, 1),
        norm_layer: Callable = nn.LayerNorm,
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
            dim_out (int): Output feature dimension
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

        self.pool_first = pool_first
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        dim_out = dim if not dim_out else dim_out
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # Set placeholders for torchscriptability, may not be actually used
        self.q = self.k = self.v = self.qkv = nn.Identity()
        if self.pool_first or self.separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out, bias=True if bias_on else False)

        if dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(dropout_rate)
        else:
            self.proj_drop = nn.Identity()

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            kernel_q is not None
            and self._prod(kernel_q) == 1
            and self._prod(stride_q) == 1
        ):
            kernel_q = None
        if (
            kernel_kv is not None
            and self._prod(kernel_kv) == 1
            and self._prod(stride_kv) == 1
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
            if self.pool_first:
                dim_conv = dim // num_heads
            else:
                dim_conv = dim_out // num_heads
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_q is not None
                else None
            )
            self.norm_q = norm_layer(dim_conv) if kernel_q is not None else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_k = norm_layer(dim_conv) if kernel_kv is not None else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv if depthwise_conv else 1,
                    bias=False,
                )
                if kernel_kv is not None
                else None
            )
            self.norm_v = norm_layer(dim_conv) if kernel_kv is not None else None
        else:
            raise NotImplementedError(f"Unsupported model {pool_mode}")

        # Will not be used if `separate_qkv == True`
        self._attention_pool_q = _AttentionPool(
            self.pool_q,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        self._attention_pool_k = _AttentionPool(
            self.pool_k,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        self._attention_pool_v = _AttentionPool(
            self.pool_v,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

    def _qkv_proj(
        self,
        q: torch.Tensor,
        q_size: int,
        k: torch.Tensor,
        k_size: int,
        v: torch.Tensor,
        v_size: int,
        batch_size: int,
        chan_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project the query (q), key (k), and value (v) tensors.

        Args:
            q (torch.Tensor): Query tensor.
            q_size (int): Query size.
            k (torch.Tensor): Key tensor.
            k_size (int): Key size.
            v (torch.Tensor): Value tensor.
            v_size (int): Value size.
            batch_size (int): Batch size.
            chan_size (int): Channel size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Projected query, key, and value tensors.

        This method applies linear projections to the query, key, and value tensors and reshapes them
        as needed for subsequent attention computations.
        """
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
        """
        Apply pooling to query (q), key (k), and value (v) tensors.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            thw_shape (List[int]): The shape of the input tensor (before flattening).

        Returns:
            Tuple[torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]]:
            Processed query, key, and value tensors along with their respective shapes.

        This method applies attention pooling to the query, key, and value tensors and returns
        the processed tensors along with their shapes.
        """

        q, q_shape = self._attention_pool_q(q, thw_shape)
        k, k_shape = self._attention_pool_k(k, thw_shape)
        v, v_shape = self._attention_pool_v(v, thw_shape)
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int, int, int]:
        """
        Calculate the lengths of query (q), key (k), and value (v) tensors.

        Args:
            q_shape (List[int]): Shape of the query tensor.
            k_shape (List[int]): Shape of the key tensor.
            v_shape (List[int]): Shape of the value tensor.

        Returns:
            Tuple[int, int, int]: Lengths of query, key, and value tensors.

        This method calculates the lengths of query, key, and value tensors, taking into account
        whether the input tensor contains a cls token.
        """
        q_N = self._prod(q_shape) + 1 if self.has_cls_embed else self._prod(q_shape)
        k_N = self._prod(k_shape) + 1 if self.has_cls_embed else self._prod(k_shape)
        v_N = self._prod(v_shape) + 1 if self.has_cls_embed else self._prod(v_shape)
        return q_N, k_N, v_N

    def _prod(self, shape: List[int]) -> int:
        """
        Torchscriptable version of `numpy.prod`. Note that `_prod([]) == 1`

        Args:
            shape (List[int]): List of dimensions.

        Returns:
            int: Product of the dimensions in the shape list.

        This method calculates the product of dimensions in the input list, equivalent to
        `numpy.prod`.
        """
        p: int = 1
        for dim in shape:
            p *= dim
        return p

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape and transpose the query (q), key (k), and value (v) tensors.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            q_N (int): Length of query tensor.
            v_N (int): Length of value tensor.
            k_N (int): Length of key tensor.
            B (int): Batch size.
            C (int): Channel size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reshaped and transposed query, key, and value tensors.

        This method reshapes and transposes the query, key, and value tensors for further computation
        in the attention mechanism.
        """
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Forward pass through the MultiScaleAttention block.

        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).

        Returns:
            Tuple[torch.Tensor, List[int]]: Output tensor and updated shape.

        This method computes the forward pass through the MultiScaleAttention block, including
        projections, pooling, attention, and transformations.
        """

        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
            q_N, k_N, v_N = self._get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self._reshape_qkv_to_seq(q, k, v, q_N, v_N, k_N, B, C)
            q, k, v = self._qkv_proj(q, q_N, k, k_N, v, v_N, B, self.dim_out)
        else:
            if self.separate_qkv:
                q = k = v = x
                q, k, v = self._qkv_proj(q, N, k, N, v, N, B, self.dim_out)
            else:
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        N = q.shape[2]

        if self.residual_pool:
            x = (attn @ v + q).transpose(1, 2).reshape(B, -1, self.dim_out)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim_out)

        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load parameters from a state dictionary with support for backward compatibility.

        Args:
            state_dict: State dictionary.
            prefix: Prefix for the keys in the state dictionary.
            local_metadata: Local metadata.
            strict: Whether to enforce strict loading.
            missing_keys: List to store missing keys.
            unexpected_keys: List to store unexpected keys.
            error_msgs: List to store error messages.

        This method is used to load parameters from a state dictionary with support for backward
        compatibility by renaming keys as needed.
        """
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for layer in ["pool", "norm"]:
                for pattern in ["q", "k", "v"]:
                    for type in ["weight", "bias"]:
                        old_key = f"{prefix}{layer}_{pattern}.{type}"
                        new_key = f"{prefix}_attention_pool_{pattern}.{layer}.{type}"
                        if old_key in state_dict:
                            state_dict[new_key] = state_dict[old_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Vision Transformer block with Multi-Scale Attention and MLP layers.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention  [Proj: dim expand]
                                  [dim expand]             Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp          [Proj: dim expand]
                                   [dim expand]             |
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+

    Args:
        dim (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        num_heads (int): Number of heads in the attention layer.
        mlp_ratio (float): MLP ratio controlling the hidden layer dimension.
        qkv_bias (bool): Whether to use bias in the QKV projection.
        dropout_rate (float): Dropout rate (0.0 by default, disabled).
        droppath_rate (float): DropPath rate (0.0 by default, disabled).
        act_layer (nn.Module): Activation layer used in the MLP block.
        norm_layer (nn.Module): Normalization layer.
        attn_norm_layer (nn.Module): Normalization layer in the attention module.
        dim_mul_in_att (bool): If True, dimension expansion occurs inside the attention module,
            otherwise it occurs in the MLP block.
        kernel_q (_size_3_t): Pooling kernel size for q (1, 1, 1 by default).
        kernel_kv (_size_3_t): Pooling kernel size for kv (1, 1, 1 by default).
        stride_q (_size_3_t): Pooling kernel stride for q (1, 1, 1 by default).
        stride_kv (_size_3_t): Pooling kernel stride for kv (1, 1, 1 by default).
        pool_mode (str): Pooling mode ("conv" by default, can be "avg", or "max").
        has_cls_embed (bool): Whether the input tensor contains a cls token.
        pool_first (bool): If True, apply pooling before qkv projection.
        residual_pool (bool): If True, use pooling with Improved Multiscale Vision Transformer's
            pooling residual connection.
        depthwise_conv (bool): Whether to use depthwise or full convolution for pooling.
        bias_on (bool): Whether to use biases for linear layers.
        separate_qkv (bool): Whether to use separate layers for qkv projections.

    This class represents a Multi-Scale Vision Transformer block, which consists of a Multi-Scale
    Attention layer and an MLP layer. The block can perform dimension expansion either inside the
    attention module or the MLP block based on the `dim_mul_in_att` parameter.

    Example:
    ```
    multi_scale_block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8)
    output_tensor, output_shape = multi_scale_block(input_tensor, [32, 16, 16])
    ```
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
        dim_mul_in_att: bool = False,
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
            dim_mul_in_att (bool): If set to True, dimension expansion happens inside
                the attention module, otherwise it happens in the Mlp block. Default: False.
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
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        self.norm1_is_batchnorm_1d = isinstance(self.norm1, nn.BatchNorm1d)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim=dim,
            dim_out=att_dim,
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
        self.norm2 = norm_layer(att_dim)
        self.norm2_is_batchnorm_1d = isinstance(self.norm2, nn.BatchNorm1d)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
            bias_on=bias_on,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out, bias=bias_on)
        else:
            self.proj = nn.Identity()

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )
        self._attention_pool = _AttentionPool(
            self.pool_skip, has_cls_embed=self.has_cls_embed, norm=None
        )

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Forward pass of the MultiScaleBlock.

        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).

        Returns:
            torch.Tensor: Output tensor.
            List[int]: Output tensor shape (before flattening).

        This method processes the input tensor through the Multi-Scale Attention and MLP layers,
        handling dimension expansion based on the configuration. It can also apply pooling if
        specified.
        """

        x_norm = (
            self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
            if self.norm1_is_batchnorm_1d
            else self.norm1(x)
        )
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = self._attention_pool(x, thw_shape)
        x = x_res + self.drop_path(x_block)
        x_norm = (
            self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
            if self.norm2_is_batchnorm_1d
            else self.norm2(x)
        )
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new

from typing import Callable, Tuple

import torch
import torch.nn as nn


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

    The default builder can be found in `create_default_bottleneck_block`.
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
        self._set_attributes(locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b, self.conv_c))
        if self.norm_c is not None:
            # This flag is used for weight initialization.
            self.norm_c.block_final_bn = True

    def _set_attributes(self, params: list = None) -> None:
        """
        Set attributes from the input list of parameters.
        Args:
            params (list): list of parameters.
        """
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

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


def create_default_bottleneck_block(
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
            conv_b_num_groups (int): number of groups for groupwise convolution for conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        BN related configs:
            norm (callable): a callable that constructs normalization layer, examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer, examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): resnet default bottleneck block.
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
    act_b = (None if activation is None else activation())

    conv_c = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_out,
        kernel_size=(1, 1, 1),
        bias=False,
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

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.models.utils import set_attributes


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
            conv_b_num_groups (int): number of groups for groupwise convolution for
                conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        Normalization related configs:
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
    act_b = None if activation is None else activation()

    conv_c = nn.Conv3d(
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


class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The default builder can be found in `create_default_res_block`.
    """

    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch1_norm: nn.Module = None,
        branch2: nn.Module = None,
        activation: nn.Module = None,
    ) -> nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = x + self.branch2(x)
        else:
            residual = self.branch1_conv(x)
            if self.branch1_norm is not None:
                residual = self.branch1_norm(residual)
            x = residual + self.branch2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def create_default_res_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable,
    use_shortcut: bool = True,
    # Conv configs.
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
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

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
        Bottleneck block related configs:
            dim_in (int): input channel size to the bottleneck block.
            dim_inner (int): intermediate channel size of the bottleneck.
            dim_out (int): output channel size of the bottleneck.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_default_bottleneck_block.

        Convolution related configs:
            conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            conv_a_stride (tuple): convolutional stride size(s) for conv_a.
            conv_a_padding (tuple): convolutional padding(s) for conv_a.
            conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            conv_b_stride (tuple): convolutional stride size(s) for conv_b.
            conv_b_padding (tuple): convolutional padding(s) for conv_b.
            conv_b_num_groups (int): number of groups for groupwise convolution for
                conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        Normalization related configs:
            norm (callable): a callable that constructs normalization layer. Examples
                include nn.BatchNorm3d, None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer. Examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): resnet basic block layer.
    """
    norm_model = None
    if norm is not None and dim_in != dim_out:
        norm_model = norm(num_features=dim_out)

    return ResBlock(
        branch1_conv=nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=tuple(map(np.prod, zip(conv_a_stride, conv_b_stride))),
        )
        if dim_in != dim_out and use_shortcut
        else None,
        branch1_norm=norm_model if dim_in != dim_out and use_shortcut else None,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride,
            conv_a_padding=conv_a_padding,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride,
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation,
        ),
        activation=None if activation is None else activation(),
    )


class ResStage(nn.Module):
    """
    ResStage composes sequential blocks that make up a ResNet. These blocks could be,
    for example, Residual blocks, Non-Local layers, or Squeeze-Excitation layers.

                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                      (Optional)
                                      Plugin layer
                                           ↓
                                          ...
                                           ↓
                                       ResBlock
                                           ↓
                                      (Optional)
                                      Plugin layer

    The default builder can be found in `create_default_res_stage`.
    """

    def __init__(self, res_blocks: List[nn.Module] = None) -> nn.Module:
        """
        Args:
            res_blocks (list of torch.nn.modules): a list of ResBlock module(s).
        """
        super().__init__()
        self._construct_model(res_blocks)

    def _construct_model(self, res_blocks: List[nn.Module] = None) -> None:
        """
        Constructs a nn.ModuleList model from `res_blocks`.
            res_blocks (list of torch.nn.modules): a list of ResBlock module(s).
        """
        models = []
        for ind in range(len(res_blocks)):
            models.append(res_blocks[ind])
        self.stage = torch.nn.ModuleList(models)

    def forward(self, x) -> torch.Tensor:
        for _, m in enumerate(self.stage):
            x = m(x)
        return x


def create_default_res_stage(
    *,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable,
    # Conv configs.
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
    Create default Residual Stage, which composes sequential blocks that make up a
    ResNet. These blocks could be, for example, Residual blocks, Non-Local layers, or
    Squeeze-Excitation layers.


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                          ...
                                           ↓
                                       ResBlock

    Normalization examples include: BatchNorm3d and None (no normalization).
    Activation examples include: ReLU, Softmax, Sigmoid, and None (no activation).
    Bottleneck examples include: create_default_bottleneck_block.

    Args:
        Bottleneck block related configs:
            dim_in (int): input channel size to the bottleneck block.
            dim_inner (int): intermediate channel size of the bottleneck.
            dim_out (int): output channel size of the bottleneck.
            bottleneck (callable): a callable that constructs bottleneck block layer.
                Examples include: create_default_bottleneck_block.

        Convolution related configs:
            conv_a_kernel_size (tuple): convolutional kernel size(s) for conv_a.
            conv_a_stride (tuple): convolutional stride size(s) for conv_a.
            conv_a_padding (tuple): convolutional padding(s) for conv_a.
            conv_b_kernel_size (tuple): convolutional kernel size(s) for conv_b.
            conv_b_stride (tuple): convolutional stride size(s) for conv_b.
            conv_b_padding (tuple): convolutional padding(s) for conv_b.
            conv_b_num_groups (int): number of groups for groupwise convolution for
                conv_b.
            conv_b_dilation (tuple): dilation for 3D convolution for conv_b.

        BN related configs:
            norm (callable): a callable that constructs normalization layer. Examples
                include nn.BatchNorm3d, and None (not performing normalization).
            norm_eps (float): normalization epsilon.
            norm_momentum (float): normalization momentum.

        Activation related configs:
            activation (callable): a callable that constructs activation layer. Examples
                include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
                activation).

    Returns:
        (nn.Module): resnet basic stage layer.
    """
    res_blocks = []
    for ind in range(depth):
        block = create_default_res_block(
            dim_in=dim_in if ind == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride if ind == 0 else (1, 1, 1),
            conv_a_padding=conv_a_padding,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride if ind == 0 else (1, 1, 1),
            conv_b_padding=conv_b_padding,
            conv_b_num_groups=conv_b_num_groups,
            conv_b_dilation=conv_b_dilation,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation,
        )
        res_blocks.append(block)
    return ResStage(res_blocks=res_blocks)

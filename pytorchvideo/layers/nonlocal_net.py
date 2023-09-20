# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes


class NonLocal(nn.Module):
    """
    Implementation of Non-local Neural Networks, which capture long-range dependencies
    in feature maps. Non-local Network computes the response at a position as a weighted
    sum of the features at all positions This building block can be integrated into 
    various computer vision architectures.

    Reference:
    Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    In Proceedings of the IEEE conference on CVPR, 2018.

    Args:
        conv_theta (nn.Module): Convolutional layer for computing the 'theta' transformation.
        conv_phi (nn.Module): Convolutional layer for computing the 'phi' transformation.
        conv_g (nn.Module): Convolutional layer for computing the 'g' transformation.
        conv_out (nn.Module): Convolutional layer for the output transformation.
        pool (Optional[nn.Module]): Optional temporal-spatial pooling layer to reduce computation.
        norm (Optional[nn.Module]): Optional normalization layer to be applied to the output.
        instantiation (str): The type of normalization used. Options are 'dot_product' and 'softmax'.

    Note:
        - The 'conv_theta', 'conv_phi', 'conv_g', and 'conv_out' modules should have
          matching output and input dimensions.
        - 'pool' can be used for temporal-spatial pooling to reduce computation.
        - 'instantiation' determines the type of normalization applied to the affinity tensor.

    Example:
        To create a Non-local block:
        ```
        non_local_block = NonLocal(
            conv_theta=nn.Conv3d(in_channels, inner_channels, kernel_size=1),
            conv_phi=nn.Conv3d(in_channels, inner_channels, kernel_size=1),
            conv_g=nn.Conv3d(in_channels, inner_channels, kernel_size=1),
            conv_out=nn.Conv3d(inner_channels, in_channels, kernel_size=1),
            pool=nn.MaxPool3d(kernel_size=(1, 2, 2)),
            norm=nn.BatchNorm3d(inner_channels),
            instantiation='dot_product'
        )
        ```

    Returns:
        torch.Tensor: The output tensor with long-range dependencies captured.
    """

    def __init__(
        self,
        *,
        conv_theta: nn.Module,
        conv_phi: nn.Module,
        conv_g: nn.Module,
        conv_out: nn.Module,
        pool: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        instantiation: str = "dot_product",
    ) -> None:
        super().__init__()
        set_attributes(self, locals())
        assert None not in (conv_theta, conv_phi, conv_g, conv_out)
        assert instantiation in (
            "dot_product",
            "softmax",
        ), "Unknown norm type {}".format(instantiation)
        assert (
            len(
                {
                    self.conv_theta.out_channels,
                    self.conv_phi.out_channels,
                    self.conv_g.out_channels,
                    self.conv_out.in_channels,
                }
            )
            == 1
        ), "Nonlocal convolution's input/ output dimension mismatch."

    def forward(self, x) -> torch.Tensor:
        dim_inner = self.conv_theta.out_channels

        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)
        # Perform temporal-spatial pooling to reduce the computation.
        if self.pool is not None:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, dim_inner, -1)
        phi = phi.view(N, dim_inner, -1)
        g = g.view(N, dim_inner, -1)

        # (N, C, TxHxW) x (N, C, TxHxW) => (N, TxHxW, TxHxW).
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (dim_inner**-0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim

        # (N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))
        # (N, C, TxHxW) => (N, C, T, H, W).
        theta_phi_g = theta_phi_g.view(N, dim_inner, T, H, W)
        p = self.conv_out(theta_phi_g)
        if self.norm is not None:
            p = self.norm(p)
        return x_identity + p


def create_nonlocal(
    *,
    # Nonlocal configs.
    dim_in: int,
    dim_inner: int,
    pool_size: Optional[Tuple[int]] = (1, 1, 1),
    instantiation: str = "softmax",
    # Norm configs.
    norm: Optional[Callable] = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
):
    """
    Create a Non-local Neural Network block for capturing long-range dependencies in computer
    vision architectures.Non-local Network computes the response at a position as a weighted
    sum of the features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971

    Args:
        dim_in (int): The number of dimensions for the input.
        dim_inner (int): The number of dimensions inside the Non-local block.
        pool_size (tuple[int]): The kernel size of spatial-temporal pooling. It consists of
            three integers representing the temporal pool kernel size, spatial pool kernel
            width, and spatial pool kernel height, respectively. If set to (1, 1, 1), no
            pooling is used. Default is (1, 1, 1).
        instantiation (string): The instantiation method for normalizing the correlation
            matrix. Supports two options: "dot_product" (normalizing correlation matrix
            with L2) and "softmax" (normalizing correlation matrix with Softmax).
        norm (nn.Module): An instance of nn.Module for the normalization layer. Default is
            nn.BatchNorm3d.
        norm_eps (float): The normalization epsilon.
        norm_momentum (float): The normalization momentum.

    Returns:
        NonLocal: A Non-local Neural Network block that can be integrated into computer
        vision architectures.

    Example:
        To create a Non-local block with a temporal pool size of 2x2x2 and "dot_product"
        instantiation:
        ```
        non_local_block = create_nonlocal(
            dim_in=256,
            dim_inner=128,
            pool_size=(2, 2, 2),
            instantiation="dot_product",
            norm=nn.BatchNorm3d,
            norm_eps=1e-5,
            norm_momentum=0.1
        )
        ```

    Note:
        - The Non-local block is a useful building block for capturing long-range
          dependencies in computer vision tasks.
        - You can customize the architecture of the Non-local block by specifying the
          input dimension (`dim_in`), inner dimension (`dim_inner`), pooling size
          (`pool_size`), and normalization settings (`norm`, `norm_eps`, and `norm_momentum`).
    """
          
    if pool_size is None:
        pool_size = (1, 1, 1)
    assert isinstance(pool_size, Iterable)

    if norm is None:
        norm_model = None
    else:
        norm_model = norm(num_features=dim_in, eps=norm_eps, momentum=norm_momentum)

    if any(size > 1 for size in pool_size):
        pool_model = nn.MaxPool3d(
            kernel_size=pool_size, stride=pool_size, padding=[0, 0, 0]
        )
    else:
        pool_model = None

    return NonLocal(
        conv_theta=nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0),
        conv_phi=nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0),
        conv_g=nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0),
        conv_out=nn.Conv3d(dim_inner, dim_in, kernel_size=1, stride=1, padding=0),
        pool=pool_model,
        norm=norm_model,
        instantiation=instantiation,
    )

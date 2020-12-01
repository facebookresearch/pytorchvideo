# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from pytorchvideo.layers.utils import set_attributes


class NonLocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper:
    Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    In Proceedings of the IEEE conference on CVPR, 2018.
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
            theta_phi = theta_phi * (dim_inner ** -0.5)
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
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971
    Args:
        dim_in (int): number of dimension for the input.
        dim_inner (int): number of dimension inside of the Non-local block.
        pool_size (tuple[int]): the kernel size of spatial temporal pooling,
            temporal pool kernel size, spatial pool kernel size, spatial pool kernel
            size in order. By default pool_size is None, then there would be no pooling
            used.
        instantiation (string): supports two different instantiation method:
            "dot_product": normalizing correlation matrix with L2.
            "softmax": normalizing correlation matrix with Softmax.
        norm (nn.Module): nn.Module for the normalization layer. The default is
            nn.BatchNorm3d.
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
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

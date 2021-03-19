# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d3x1x1BnAct,
    Conv3d3x3x3DwBnAct,
    Conv3d5x1x1BnAct,
    Conv3dPwBnAct,
    Conv3dTemporalKernel1BnAct,
)


def transmute_Conv3dPwBnAct(input_module: nn.Module):
    """
    Given an input_module, transmutes it into a equivalent Conv3dPwBnAct. Returns None
    if no equivalent Conv3dPwBnAct is found, else returns an instance of equivalent
    Conv3dPwBnAct.
    Args:
        input_module (nn.Module): input module to find an equivalent Conv3dPwBnAct
    """
    if not isinstance(input_module, nn.Conv3d):
        return None
    if (
        input_module.kernel_size == (1, 1, 1)
        and input_module.groups == 1
        and input_module.stride == (1, 1, 1)
        and input_module.padding == (0, 0, 0)
        and input_module.dilation == (1, 1, 1)
    ):
        module = Conv3dPwBnAct(
            in_channels=input_module.in_channels,
            out_channels=input_module.out_channels,
            bias=False if input_module.bias is None else True,
            activation="identity",
            use_bn=False,
        )
        module.kernel.conv.load_state_dict(input_module.state_dict())
        return module
    else:
        return None


def transmute_Conv3d3x3x3DwBnAct(input_module: nn.Module):
    """
    Given an input_module, transmutes it into a equivalent Conv3d3x3x3DwBnAct. Returns
    None if no equivalent Conv3d3x3x3DwBnAct is found, else returns an instance of
    equivalent Conv3d3x3x3DwBnAct.
    Args:
        input_module (nn.Module): input module to find an equivalent Conv3d3x3x3DwBnAct
    """
    if not isinstance(input_module, nn.Conv3d):
        return None
    if (
        input_module.kernel_size == (3, 3, 3)
        and input_module.in_channels == input_module.out_channels
        and input_module.groups == input_module.out_channels
        and input_module.stride[0] == 1
        and input_module.stride[1] == input_module.stride[2]
        and input_module.padding == (1, 1, 1)
        and input_module.padding_mode == "zeros"
        and input_module.dilation == (1, 1, 1)
    ):
        spatial_stride = input_module.stride[1]
        module = Conv3d3x3x3DwBnAct(
            in_channels=input_module.in_channels,
            spatial_stride=spatial_stride,
            bias=False if input_module.bias is None else True,
            activation="identity",
            use_bn=False,
        )
        module.kernel.conv.load_state_dict(input_module.state_dict())
        return module
    else:
        return None


def transmute_Conv3dTemporalKernel1BnAct(input_module: nn.Module):
    """
    Given an input_module, transmutes it into a equivalent Conv3dTemporalKernel1BnAct.
    Returns None if no equivalent Conv3dTemporalKernel1BnAct is found, else returns
    an instance of equivalent Conv3dTemporalKernel1BnAct.
    Args:
        input_module (nn.Module): input module to find an equivalent Conv3dTemporalKernel1BnAct
    """
    if not isinstance(input_module, nn.Conv3d):
        return None
    """
    If the input_module can be replaced by Conv3dPwBnAct, don't use
    Conv3dTemporalKernel1BnAct.
    """
    if (
        input_module.kernel_size == (1, 1, 1)
        and input_module.groups == 1
        and input_module.stride == (1, 1, 1)
        and input_module.padding == (0, 0, 0)
        and input_module.dilation == (1, 1, 1)
    ):
        return None

    if (
        input_module.kernel_size[0] == 1
        and input_module.kernel_size[1] == input_module.kernel_size[2]
        and input_module.stride[0] == 1
        and input_module.stride[1] == input_module.stride[2]
        and input_module.padding[0] == 0
        and input_module.dilation[0] == 1
    ):
        spatial_stride = input_module.stride[1]
        spatial_kernel = input_module.kernel_size[1]
        spatial_padding = input_module.padding[1]
        spatial_dilation = input_module.dilation[1]
        module = Conv3dTemporalKernel1BnAct(
            in_channels=input_module.in_channels,
            out_channels=input_module.out_channels,
            bias=False if input_module.bias is None else True,
            groups=input_module.groups,
            spatial_kernel=spatial_kernel,
            spatial_stride=spatial_stride,
            spatial_padding=spatial_padding,
            spatial_dilation=spatial_dilation,
            activation="identity",
            use_bn=False,
        )
        module.kernel.conv.load_state_dict(input_module.state_dict())
        return module
    else:
        return None


def transmute_Conv3d3x1x1BnAct(input_module: nn.Module):
    """
    Given an input_module, transmutes it into a equivalent Conv3d3x1x1BnAct.
    Returns None if no equivalent Conv3d3x1x1BnAct is found, else returns
    an instance of equivalent Conv3d3x1x1BnAct.
    Args:
        input_module (nn.Module): input module to find an equivalent Conv3d3x1x1BnAct
    """
    if not isinstance(input_module, nn.Conv3d):
        return None

    if (
        input_module.kernel_size == (3, 1, 1)
        and input_module.stride == (1, 1, 1)
        and input_module.padding == (1, 0, 0)
        and input_module.dilation == (1, 1, 1)
        and input_module.padding_mode == "zeros"
    ):
        module = Conv3d3x1x1BnAct(
            in_channels=input_module.in_channels,
            out_channels=input_module.out_channels,
            bias=False if input_module.bias is None else True,
            groups=input_module.groups,
            activation="identity",
            use_bn=False,
        )
        module.kernel.conv.load_state_dict(input_module.state_dict())
        return module
    else:
        return None


def transmute_Conv3d5x1x1BnAct(input_module: nn.Module):
    """
    Given an input_module, transmutes it into a equivalent Conv3d5x1x1BnAct.
    Returns None if no equivalent Conv3d5x1x1BnAct is found, else returns
    an instance of equivalent Conv3d5x1x1BnAct.
    Args:
        input_module (nn.Module): input module to find an equivalent Conv3d5x1x1BnAct
    """
    if not isinstance(input_module, nn.Conv3d):
        return None

    if (
        input_module.kernel_size == (5, 1, 1)
        and input_module.stride == (1, 1, 1)
        and input_module.padding == (2, 0, 0)
        and input_module.dilation == (1, 1, 1)
        and input_module.padding_mode == "zeros"
    ):
        module = Conv3d5x1x1BnAct(
            in_channels=input_module.in_channels,
            out_channels=input_module.out_channels,
            bias=False if input_module.bias is None else True,
            groups=input_module.groups,
            activation="identity",
            use_bn=False,
        )
        module.kernel.conv.load_state_dict(input_module.state_dict())
        return module
    else:
        return None


"""
List of efficient_block transmuters for mobile_cpu. If one module matches multiple
transmuters, the first matched transmuter in list will be used.
"""
EFFICIENT_BLOCK_TRANSMUTER_MOBILE_CPU = [
    transmute_Conv3dPwBnAct,
    transmute_Conv3d3x3x3DwBnAct,
    transmute_Conv3dTemporalKernel1BnAct,
    transmute_Conv3d3x1x1BnAct,
    transmute_Conv3d5x1x1BnAct,
]

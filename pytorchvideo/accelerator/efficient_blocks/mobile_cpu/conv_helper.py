# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This file contains helper classes for building conv3d efficient blocks.
The helper classes are intended to be instantiated inside efficient block,
not to be used by user to build network.
"""

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn


class _Reshape(nn.Module):
    """
    Helper class to implement data reshape as a module.
    Args:
        reshape_size (tuple): size of data after reshape.
    """

    def __init__(
        self,
        reshape_size: Tuple,
    ):
        super().__init__()
        self.reshape_size = reshape_size

    def forward(self, x):
        return torch.reshape(x, self.reshape_size)


class _Conv3dTemporalKernel3Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with temporal kernel of 3 into equivalent conv2ds.
    In conv3d with temporal kernel 3 and input I, for output temporal index of t (O[:,:,t,:,:]),
    the conv can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t:t+3,:,:])
                 = conv2d_0(I[:,:,t,:,:]) + conv2d_1(I[:,:,t+1,:,:]) + conv2d_2(I[:,:,t+2,:,:])
    If bias is considered:
    O[:,:,t,:,:] = conv3d_w_bias(I[:,:,t:t+3,:,:])
                 = conv2d_0_wo_bias(I[:,:,t,:,:])
                   + conv2d_1_w_bias(I[:,:,t+1,:,:]) + conv2d_2_wo_bias(I[:,:,t+2,:,:])
    The input Conv3d also needs zero padding of size 1 in temporal dimension.
    """

    def __init__(
        self,
        conv3d_in: nn.Conv3d,
        input_THW_tuple: Tuple,
    ):
        """
        Args:
            conv3d_in (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            input_THW_tuple (tuple): input THW size for conv3d_in during forward.
        """
        super().__init__()
        assert conv3d_in.padding[0] == 1, (
            "_Conv3dTemporalKernel3Eq only support temporal padding of 1, "
            f"but got {conv3d_in.padding[0]}"
        )
        assert conv3d_in.padding_mode == "zeros", (
            "_Conv3dTemporalKernel3Eq only support zero padding, "
            f"but got {conv3d_in.padding_mode}"
        )
        self._input_THW_tuple = input_THW_tuple
        padding_2d = conv3d_in.padding[1:]
        in_channels = conv3d_in.in_channels
        out_channels = conv3d_in.out_channels
        kernel_size = conv3d_in.kernel_size[1:]
        groups = conv3d_in.groups
        stride_2d = conv3d_in.stride[1:]
        # Create 3 conv2d to emulate conv3d.
        if (
            self._input_THW_tuple[0] > 1
        ):  # Those two conv2d are needed only when temporal input > 1.
            self._conv2d_3_3_0 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding_2d,
                stride=stride_2d,
                groups=groups,
                bias=False,
            )
            self._conv2d_3_3_2 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding_2d,
                stride=stride_2d,
                groups=groups,
                bias=False,
            )
        self._conv2d_3_3_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_2d,
            stride=stride_2d,
            groups=groups,
            bias=(conv3d_in.bias is not None),
        )

        state_dict = conv3d_in.state_dict()
        state_dict_1 = deepcopy(state_dict)
        state_dict_1["weight"] = state_dict["weight"][:, :, 1]
        self._conv2d_3_3_1.load_state_dict(state_dict_1)

        if self._input_THW_tuple[0] > 1:
            state_dict_0 = deepcopy(state_dict)
            state_dict_0["weight"] = state_dict["weight"][:, :, 0]
            if conv3d_in.bias is not None:
                """
                Don't need bias for other conv2d instances to avoid duplicated addition of bias.
                """
                state_dict_0.pop("bias")
            self._conv2d_3_3_0.load_state_dict(state_dict_0)

            state_dict_2 = deepcopy(state_dict)
            state_dict_2["weight"] = state_dict["weight"][:, :, 2]
            if conv3d_in.bias is not None:
                state_dict_2.pop("bias")
            self._conv2d_3_3_2.load_state_dict(state_dict_2)

            self._add_funcs = nn.ModuleList(
                [
                    nn.quantized.FloatFunctional()
                    for _ in range(2 * (self._input_THW_tuple[0] - 1))
                ]
            )
            self._cat_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Use three conv2d to emulate conv3d.
        This forward assumes zero padding of size 1 in temporal dimension.
        """
        if self._input_THW_tuple[0] > 1:
            out_tensor_list = []
            """
            First output plane in temporal dimension,
            conv2d_3_3_0 is skipped due to zero padding.
            """
            cur_tensor = (
                self._add_funcs[0]
                .add(self._conv2d_3_3_1(x[:, :, 0]), self._conv2d_3_3_2(x[:, :, 1]))
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            for idx in range(2, self._input_THW_tuple[0]):
                cur_tensor = (
                    self._add_funcs[2 * idx - 3]
                    .add(
                        self._add_funcs[2 * idx - 2].add(
                            self._conv2d_3_3_0(x[:, :, idx - 2]),
                            self._conv2d_3_3_1(x[:, :, idx - 1]),
                        ),
                        self._conv2d_3_3_2(x[:, :, idx]),
                    )
                    .unsqueeze(2)
                )
                out_tensor_list.append(cur_tensor)
            """
            Last output plane in temporal domain, conv2d_3_3_2 is skipped due to zero padding.
            """
            cur_tensor = (
                self._add_funcs[-1]
                .add(self._conv2d_3_3_0(x[:, :, -2]), self._conv2d_3_3_1(x[:, :, -1]))
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            return self._cat_func.cat(out_tensor_list, 2)
        else:  # Degenerated to simple conv2d
            return self._conv2d_3_3_1(x[:, :, 0])

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


class _SkipConnectMul(nn.Module):
    """
    Helper class to implement skip multiplication.
    Args:
        layer (nn.Module): layer for skip multiplication. With input x, _SkipConnectMul
            implements layer(x)*x.
    """

    def __init__(
        self,
        layer: nn.Module,
    ):
        super().__init__()
        self.layer = layer
        self.mul_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.mul_func.mul(x, self.layer(x))


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
            return self._conv2d_3_3_1(x[:, :, 0]).unsqueeze(2)


class _Conv3dTemporalKernel5Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with kernel size of (5, k, k) into equivalent conv2ds.
    In such conv3d and input I, for output temporal index of t (O[:,:,t,:,:]), the conv
    can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t:t+5,:,:])
                 = conv2d_0(I[:,:,t,:,:]) + conv2d_1(I[:,:,t+1,:,:]) + conv2d_2(I[:,:,t+2,:,:])
                   + conv2d_3(I[:,:,t+3,:,:]) + conv2d_4(I[:,:,t+4,:,:])
    If bias is considered:
    O[:,:,t,:,:] = conv3d_w_bias(I[:,:,t:t+3,:,:])
                 = conv2d_0_wo_bias(I[:,:,t,:,:])
                   + conv2d_1_wo_bias(I[:,:,t+1,:,:]) + conv2d_2_w_bias(I[:,:,t+2,:,:])
                   + conv2d_3_wo_bias(I[:,:,t+1,:,:]) + conv2d_4_wo_bias(I[:,:,t+2,:,:])
    The input Conv3d also needs zero padding of size 2 in temporal dimension at begin and end.
    """

    def __init__(
        self,
        conv3d_in: nn.Conv3d,
        thw_shape: Tuple[int, int, int],
    ):
        """
        Args:
            conv3d_in (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            thw_shape (tuple): input THW size for conv3d_in during forward.
        """
        super().__init__()
        assert conv3d_in.padding[0] == 2, (
            "_Conv3dTemporalKernel5Eq only support temporal padding of 2, "
            f"but got {conv3d_in.padding[0]}"
        )
        assert conv3d_in.padding_mode == "zeros", (
            "_Conv3dTemporalKernel5Eq only support zero padding, "
            f"but got {conv3d_in.padding_mode}"
        )
        self._thw_shape = thw_shape
        padding_2d = conv3d_in.padding[1:]
        in_channels = conv3d_in.in_channels
        out_channels = conv3d_in.out_channels
        kernel_size = conv3d_in.kernel_size[1:]
        groups = conv3d_in.groups
        stride_2d = conv3d_in.stride[1:]
        # Create 3 conv2d to emulate conv3d.
        t, h, w = self._thw_shape
        args_dict = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "padding": padding_2d,
            "stride": stride_2d,
            "groups": groups,
        }

        for iter_idx in range(5):
            if iter_idx != 2:
                if t > 1:  # Those four conv2d are needed only when temporal input > 1.
                    self.add_module(
                        f"_conv2d_{iter_idx}", nn.Conv2d(**args_dict, bias=False)
                    )
            else:  # _conv2d_2 is needed for all circumstances.
                self.add_module(
                    f"_conv2d_{iter_idx}",
                    nn.Conv2d(**args_dict, bias=(conv3d_in.bias is not None)),
                )

        # State dict for _conv2d_2
        original_state_dict = conv3d_in.state_dict()
        state_dict_to_load = deepcopy(original_state_dict)
        state_dict_to_load["weight"] = original_state_dict["weight"][:, :, 2]
        self._conv2d_2.load_state_dict(state_dict_to_load)

        if t > 1:
            if conv3d_in.bias is not None:
                # Don't need bias for other conv2d instances to avoid duplicated
                # addition of bias.
                state_dict_to_load.pop("bias")
            # State dict for _conv2d_0, _conv2d_1, _conv2d_3, _conv2d_4
            state_dict_to_load["weight"] = original_state_dict["weight"][:, :, 0]
            self._conv2d_0.load_state_dict(state_dict_to_load)

            state_dict_to_load["weight"] = original_state_dict["weight"][:, :, 1]
            self._conv2d_1.load_state_dict(state_dict_to_load)

            state_dict_to_load["weight"] = original_state_dict["weight"][:, :, 3]
            self._conv2d_3.load_state_dict(state_dict_to_load)

            state_dict_to_load["weight"] = original_state_dict["weight"][:, :, 4]
            self._conv2d_4.load_state_dict(state_dict_to_load)
            # Elementwise add are needed in forward function, use nn.quantized.FloatFunctional()
            # for better quantization support. One convolution needs at most 4 elementwise adds
            # without zero padding; for boundary planes fewer elementwise adds are needed.
            # See forward() for more details.
            self._add_funcs = nn.ModuleList(
                [nn.quantized.FloatFunctional() for _ in range(4 * t - 6)]
            )
            self._cat_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Use three conv2d to emulate conv3d.
        Args:
           x (torch.Tensor): 5D tensor of (B, C, T, H, W)
        """
        t, h, w = self._thw_shape
        out_tensor_list = []
        if (
            t == 1
        ):  # Degenerated to simple conv2d, but make sure output still has T dimension
            return self._conv2d_2(x[:, :, 0]).unsqueeze(2)
        elif t == 2:
            # out_tensor_list[0]: conv2d_1_1_0, conv2d_1_1_1 and conv2d_1_1_4 are
            # applied to zero padding.
            cur_tensor = (
                self._add_funcs[0]
                .add(self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1]))
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[1]: conv2d_1_1_0, conv2d_1_1_3 and conv2d_1_1_4 are
            # applied to zero padding.

            cur_tensor = (
                self._add_funcs[1]
                .add(self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1]))
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
        elif t == 3:
            # out_tensor_list[0]: conv2d_1_1_0, conv2d_1_1_1 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[0]
                .add(
                    self._add_funcs[1].add(
                        self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])
                    ),
                    self._conv2d_4(x[:, :, 2]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[1]: conv2d_1_1_0, conv2d_1_1_4 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[2]
                .add(
                    self._add_funcs[3].add(
                        self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1])
                    ),
                    self._conv2d_3(x[:, :, 2]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[2]: conv2d_1_1_3, conv2d_1_1_4 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[4]
                .add(
                    self._add_funcs[5].add(
                        self._conv2d_0(x[:, :, 0]), self._conv2d_1(x[:, :, 1])
                    ),
                    self._conv2d_2(x[:, :, 2]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
        elif t == 4:
            # out_tensor_list[0]: conv2d_1_1_0, conv2d_1_1_1 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[0]
                .add(
                    self._add_funcs[1].add(
                        self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])
                    ),
                    self._conv2d_4(x[:, :, 2]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[1]: conv2d_1_1_0 is applied to zero padding.
            cur_tensor = (
                self._add_funcs[2]
                .add(
                    self._add_funcs[3].add(
                        self._add_funcs[4].add(
                            self._conv2d_1(x[:, :, 0]),
                            self._conv2d_2(x[:, :, 1]),
                        ),
                        self._conv2d_3(x[:, :, 2]),
                    ),
                    self._conv2d_4(x[:, :, 3]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[2]: conv2d_1_1_4 is applied to zero padding.
            cur_tensor = (
                self._add_funcs[5]
                .add(
                    self._add_funcs[6].add(
                        self._add_funcs[7].add(
                            self._conv2d_0(x[:, :, 0]),
                            self._conv2d_1(x[:, :, 1]),
                        ),
                        self._conv2d_2(x[:, :, 2]),
                    ),
                    self._conv2d_3(x[:, :, 3]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            # out_tensor_list[3]: conv2d_1_1_3, conv2d_1_1_4 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[8]
                .add(
                    self._add_funcs[9].add(
                        self._conv2d_0(x[:, :, 1]), self._conv2d_1(x[:, :, 2])
                    ),
                    self._conv2d_2(x[:, :, 3]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
        else:  # t >= 5
            # out_tensor_list[0]: conv2d_1_1_0, conv2d_1_1_1 are applied to zero padding.
            add_func_idx_base = 0
            cur_tensor = (
                self._add_funcs[add_func_idx_base]
                .add(
                    self._add_funcs[add_func_idx_base + 1].add(
                        self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])
                    ),
                    self._conv2d_4(x[:, :, 2]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 2
            # out_tensor_list[1]: conv2d_1_1_0 is applied to zero padding.
            cur_tensor = (
                self._add_funcs[add_func_idx_base]
                .add(
                    self._add_funcs[add_func_idx_base + 1].add(
                        self._add_funcs[add_func_idx_base + 2].add(
                            self._conv2d_1(x[:, :, 0]),
                            self._conv2d_2(x[:, :, 1]),
                        ),
                        self._conv2d_3(x[:, :, 2]),
                    ),
                    self._conv2d_4(x[:, :, 3]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 3
            # out_tensor_list[2:-2]: zero padding has no effect.
            for idx in range(4, t):
                cur_tensor = (
                    self._add_funcs[add_func_idx_base]
                    .add(
                        self._add_funcs[add_func_idx_base + 1].add(
                            self._add_funcs[add_func_idx_base + 2].add(
                                self._add_funcs[add_func_idx_base + 3].add(
                                    self._conv2d_0(x[:, :, idx - 4]),
                                    self._conv2d_1(x[:, :, idx - 3]),
                                ),
                                self._conv2d_2(x[:, :, idx - 2]),
                            ),
                            self._conv2d_3(x[:, :, idx - 1]),
                        ),
                        self._conv2d_4(x[:, :, idx]),
                    )
                    .unsqueeze(2)
                )
                out_tensor_list.append(cur_tensor)
                add_func_idx_base += 4
            # out_tensor_list[-2]: conv2d_1_1_4 is applied to zero padding.
            cur_tensor = (
                self._add_funcs[add_func_idx_base]
                .add(
                    self._add_funcs[add_func_idx_base + 1].add(
                        self._add_funcs[add_func_idx_base + 2].add(
                            self._conv2d_0(x[:, :, -4]),
                            self._conv2d_1(x[:, :, -3]),
                        ),
                        self._conv2d_2(x[:, :, -2]),
                    ),
                    self._conv2d_3(x[:, :, -1]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 3
            # out_tensor_list[-1]: conv2d_1_1_3, conv2d_1_1_4 are applied to zero padding.
            cur_tensor = (
                self._add_funcs[add_func_idx_base]
                .add(
                    self._add_funcs[add_func_idx_base + 1].add(
                        self._conv2d_0(x[:, :, -3]),
                        self._conv2d_1(x[:, :, -2]),
                    ),
                    self._conv2d_2(x[:, :, -1]),
                )
                .unsqueeze(2)
            )
            out_tensor_list.append(cur_tensor)
        return self._cat_func.cat(out_tensor_list, 2)


class _Conv3dTemporalKernel1Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with temporal kernel of 1 into conv2d on
    multiple temporal planes.
    In conv3d with temporal kernel 1 and input I, for output temporal index of t (O[:,:,t,:,:]),
    the conv can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t,:,:])
                 = conv2d(I[:,:,t,:,:])
    The full output can be obtained by concat O[:,:,t,:,:] for t in 0...T,
    where T is the length of I in temporal dimension.
    """

    def __init__(
        self,
        conv3d_eq: nn.Conv3d,
        input_THW_tuple: Tuple,
    ):
        """
        Args:
            conv3d_eq (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            input_THW_tuple (tuple): input THW size for conv3d_eq during forward.
        """
        super().__init__()
        # create equivalent conv2d module
        in_channels = conv3d_eq.in_channels
        out_channels = conv3d_eq.out_channels
        bias_flag = conv3d_eq.bias is not None
        self.conv2d_eq = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(conv3d_eq.kernel_size[1], conv3d_eq.kernel_size[2]),
            stride=(conv3d_eq.stride[1], conv3d_eq.stride[2]),
            groups=conv3d_eq.groups,
            bias=bias_flag,
            padding=(conv3d_eq.padding[1], conv3d_eq.padding[2]),
            dilation=(conv3d_eq.dilation[1], conv3d_eq.dilation[2]),
        )
        state_dict = conv3d_eq.state_dict()
        state_dict["weight"] = state_dict["weight"].squeeze(2)
        self.conv2d_eq.load_state_dict(state_dict)
        self.input_THW_tuple = input_THW_tuple

    def forward(self, x):
        out_tensor_list = []
        for idx in range(self.input_THW_tuple[0]):
            cur_tensor = self.conv2d_eq(x[:, :, idx]).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        return torch.cat(out_tensor_list, 2)

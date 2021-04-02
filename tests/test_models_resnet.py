# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import unittest

import numpy as np
import torch
from pytorchvideo.models.head import ResNetBasicHead
from pytorchvideo.models.net import Net
from pytorchvideo.models.resnet import (
    BottleneckBlock,
    ResBlock,
    ResStage,
    SeparableBottleneckBlock,
    create_acoustic_bottleneck_block,
    create_acoustic_building_block,
    create_acoustic_resnet,
    create_bottleneck_block,
    create_res_block,
    create_res_stage,
    create_resnet,
)
from pytorchvideo.models.stem import ResNetBasicStem
from torch import nn


class TestBottleneckBlock(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_simple_bottleneck_block(self):
        """
        Test simple BottleneckBlock with different dimensions.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = BottleneckBlock(
                conv_a=nn.Conv3d(
                    dim_in, dim_inner, kernel_size=1, stride=1, padding=0, bias=False
                ),
                norm_a=nn.BatchNorm3d(dim_inner),
                act_a=nn.ReLU(),
                conv_b=nn.Conv3d(
                    dim_inner, dim_inner, kernel_size=3, stride=1, padding=1, bias=False
                ),
                norm_b=nn.BatchNorm3d(dim_inner),
                act_b=nn.ReLU(),
                conv_c=nn.Conv3d(
                    dim_inner, dim_out, kernel_size=1, stride=1, padding=0, bias=False
                ),
                norm_c=nn.BatchNorm3d(dim_out),
            )

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(input_tensor)
                    continue

                output_tensor = model(input_tensor)
                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    input_shape[2],
                    input_shape[3],
                    input_shape[4],
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_complex_bottleneck_block(self):
        """
        Test complex BottleneckBlock with different dimensions.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = BottleneckBlock(
                conv_a=nn.Conv3d(
                    dim_in,
                    dim_inner,
                    kernel_size=[3, 1, 1],
                    stride=[2, 1, 1],
                    padding=[1, 0, 0],
                    bias=False,
                ),
                norm_a=nn.BatchNorm3d(dim_inner),
                act_a=nn.ReLU(),
                conv_b=nn.Conv3d(
                    dim_inner,
                    dim_inner,
                    kernel_size=[1, 3, 3],
                    stride=[1, 2, 2],
                    padding=[0, 1, 1],
                    groups=1,
                    dilation=[1, 1, 1],
                    bias=False,
                ),
                norm_b=nn.BatchNorm3d(dim_inner),
                act_b=nn.ReLU(),
                conv_c=nn.Conv3d(
                    dim_inner,
                    dim_out,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=nn.BatchNorm3d(dim_out),
            )

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(Exception):
                        output_tensor = model(input_tensor)
                    continue

                output_tensor = model(input_tensor)
                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    (input_shape[2] - 1) // 2 + 1,
                    (input_shape[3] - 1) // 2 + 1,
                    (input_shape[4] - 1) // 2 + 1,
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_separable_bottleneck_block_sum(self):
        """
        Test SeparableBottleneckBlock with different dimensions.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = SeparableBottleneckBlock(
                conv_a=nn.Conv3d(
                    dim_in,
                    dim_inner,
                    kernel_size=[3, 1, 1],
                    stride=[2, 1, 1],
                    padding=[1, 0, 0],
                    bias=False,
                ),
                norm_a=nn.BatchNorm3d(dim_inner),
                act_a=nn.ReLU(),
                conv_b=nn.ModuleList(
                    [
                        nn.Conv3d(
                            dim_inner,
                            dim_inner,
                            kernel_size=[1, 3, 3],
                            stride=[1, 2, 2],
                            padding=[0, 1, 1],
                            groups=1,
                            dilation=[1, 1, 1],
                            bias=False,
                        ),
                        nn.Conv3d(
                            dim_inner,
                            dim_inner,
                            kernel_size=[1, 3, 3],
                            stride=[1, 2, 2],
                            padding=[0, 1, 1],
                            groups=1,
                            dilation=[1, 1, 1],
                            bias=False,
                        ),
                    ]
                ),
                norm_b=nn.ModuleList(
                    [nn.BatchNorm3d(dim_inner), nn.BatchNorm3d(dim_inner)]
                ),
                act_b=nn.ModuleList([nn.ReLU(), nn.ReLU()]),
                conv_c=nn.Conv3d(
                    dim_inner,
                    dim_out,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=nn.BatchNorm3d(dim_out),
                reduce_method="sum",
            )

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(Exception):
                        output_tensor = model(input_tensor)
                    continue

                output_tensor = model(input_tensor)
                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    (input_shape[2] - 1) // 2 + 1,
                    (input_shape[3] - 1) // 2 + 1,
                    (input_shape[4] - 1) // 2 + 1,
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_separable_complex_bottleneck_block_cat(self):
        """
        Test SeparableBottleneckBlock with different dimensions.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = SeparableBottleneckBlock(
                conv_a=nn.Conv3d(
                    dim_in,
                    dim_inner,
                    kernel_size=[3, 1, 1],
                    stride=[2, 1, 1],
                    padding=[1, 0, 0],
                    bias=False,
                ),
                norm_a=nn.BatchNorm3d(dim_inner),
                act_a=nn.ReLU(),
                conv_b=nn.ModuleList(
                    [
                        nn.Conv3d(
                            dim_inner,
                            dim_inner,
                            kernel_size=[1, 3, 3],
                            stride=[1, 2, 2],
                            padding=[0, 1, 1],
                            groups=1,
                            dilation=[1, 1, 1],
                            bias=False,
                        ),
                        nn.Conv3d(
                            dim_inner,
                            dim_inner,
                            kernel_size=[1, 3, 3],
                            stride=[1, 2, 2],
                            padding=[0, 1, 1],
                            groups=1,
                            dilation=[1, 1, 1],
                            bias=False,
                        ),
                    ]
                ),
                norm_b=nn.ModuleList(
                    [nn.BatchNorm3d(dim_inner), nn.BatchNorm3d(dim_inner)]
                ),
                act_b=nn.ModuleList([nn.ReLU(), nn.ReLU()]),
                conv_c=nn.Conv3d(
                    dim_inner * 2,
                    dim_out,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=nn.BatchNorm3d(dim_out),
                reduce_method="cat",
            )

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(Exception):
                        output_tensor = model(input_tensor)
                    continue

                output_tensor = model(input_tensor)
                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    (input_shape[2] - 1) // 2 + 1,
                    (input_shape[3] - 1) // 2 + 1,
                    (input_shape[4] - 1) // 2 + 1,
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_acoustic_bottleneck_block_with_callable(self):
        """
        Test builder `create_acoustic_bottleneck_block` with callable inputs.
        """
        for (norm_model, act_model) in itertools.product(
            (nn.BatchNorm3d,), (nn.ReLU, nn.Softmax, nn.Sigmoid)
        ):
            model = create_acoustic_bottleneck_block(
                dim_in=32,
                dim_inner=16,
                dim_out=64,
                conv_a_kernel_size=(3, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(1, 0, 0),
                conv_b_kernel_size=(3, 3, 3),
                conv_b_stride=(1, 1, 1),
                conv_b_padding=(1, 1, 1),
                conv_b_num_groups=1,
                conv_b_dilation=(1, 1, 1),
                norm=norm_model,
                activation=act_model,
            )
            model_gt = SeparableBottleneckBlock(
                conv_a=nn.Conv3d(
                    32,
                    16,
                    kernel_size=[3, 1, 1],
                    stride=[1, 1, 1],
                    padding=[1, 0, 0],
                    bias=False,
                ),
                norm_a=norm_model(16),
                act_a=act_model(),
                conv_b=nn.ModuleList(
                    [
                        nn.Conv3d(
                            16,
                            16,
                            kernel_size=[1, 3, 3],
                            stride=[1, 1, 1],
                            padding=[0, 1, 1],
                            dilation=1,
                            bias=False,
                        ),
                        nn.Conv3d(
                            16,
                            16,
                            kernel_size=[3, 1, 1],
                            stride=[1, 1, 1],
                            padding=[1, 0, 0],
                            dilation=1,
                            bias=False,
                        ),
                    ]
                ),
                norm_b=nn.ModuleList([norm_model(16), norm_model(16)]),
                act_b=nn.ModuleList([act_model(), act_model()]),
                conv_c=nn.Conv3d(
                    16,
                    64,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=norm_model(64),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in=32):
                with torch.no_grad():
                    if input_tensor.shape[1] != 32:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue

                    output_tensor = model(input_tensor)
                    output_tensor_gt = model_gt(input_tensor)
                self.assertEqual(
                    output_tensor.shape,
                    output_tensor_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        output_tensor.shape, output_tensor_gt.shape
                    ),
                )
                self.assertTrue(
                    np.allclose(output_tensor.numpy(), output_tensor_gt.numpy())
                )

    def test_create_acoustic_building_block_with_callable(self):
        """
        Test builder `create_building_bottleneck_block` with callable inputs.
        """
        for (norm_model, act_model) in itertools.product(
            (nn.BatchNorm3d,), (nn.ReLU, nn.Softmax, nn.Sigmoid)
        ):
            model = create_acoustic_building_block(
                dim_in=32,
                dim_inner=16,
                dim_out=64,
                conv_a_kernel_size=(3, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(1, 0, 0),
                conv_b_kernel_size=(3, 3, 3),
                conv_b_stride=(1, 1, 1),
                conv_b_padding=(1, 1, 1),
                conv_b_num_groups=1,
                conv_b_dilation=(1, 1, 1),
                norm=norm_model,
                activation=act_model,
            )
            model_gt = SeparableBottleneckBlock(
                conv_a=None,
                norm_a=None,
                act_a=None,
                conv_b=nn.ModuleList(
                    [
                        nn.Conv3d(
                            32,
                            16,
                            kernel_size=[3, 1, 1],
                            stride=[1, 1, 1],
                            padding=[1, 0, 0],
                            dilation=1,
                            bias=False,
                        ),
                        nn.Conv3d(
                            32,
                            16,
                            kernel_size=[1, 3, 3],
                            stride=[1, 1, 1],
                            padding=[0, 1, 1],
                            dilation=1,
                            bias=False,
                        ),
                    ]
                ),
                norm_b=nn.ModuleList([norm_model(16), norm_model(16)]),
                act_b=nn.ModuleList([act_model(), act_model()]),
                conv_c=nn.Conv3d(
                    16,
                    64,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=norm_model(64),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in=32):
                with torch.no_grad():
                    if input_tensor.shape[1] != 32:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue

                    output_tensor = model(input_tensor)
                    output_tensor_gt = model_gt(input_tensor)
                self.assertEqual(
                    output_tensor.shape,
                    output_tensor_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        output_tensor.shape, output_tensor_gt.shape
                    ),
                )
                self.assertTrue(
                    np.allclose(output_tensor.numpy(), output_tensor_gt.numpy())
                )

    def test_create_bottleneck_block_with_callable(self):
        """
        Test builder `create_bottleneck_block` with callable inputs.
        """
        for (norm_model, act_model) in itertools.product(
            (nn.BatchNorm3d,), (nn.ReLU, nn.Softmax, nn.Sigmoid)
        ):
            model = create_bottleneck_block(
                dim_in=32,
                dim_inner=16,
                dim_out=64,
                conv_a_kernel_size=(3, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(1, 0, 0),
                conv_b_kernel_size=(1, 3, 3),
                conv_b_stride=(1, 1, 1),
                conv_b_padding=(0, 1, 1),
                conv_b_num_groups=1,
                conv_b_dilation=(1, 1, 1),
                norm=norm_model,
                activation=act_model,
            )
            model_gt = BottleneckBlock(
                conv_a=nn.Conv3d(
                    32,
                    16,
                    kernel_size=[3, 1, 1],
                    stride=[1, 1, 1],
                    padding=[1, 0, 0],
                    bias=False,
                ),
                norm_a=norm_model(16),
                act_a=act_model(),
                conv_b=nn.Conv3d(
                    16,
                    16,
                    kernel_size=[1, 3, 3],
                    stride=[1, 1, 1],
                    padding=[0, 1, 1],
                    bias=False,
                ),
                norm_b=norm_model(16),
                act_b=act_model(),
                conv_c=nn.Conv3d(
                    16,
                    64,
                    kernel_size=[1, 1, 1],
                    stride=[1, 1, 1],
                    padding=[0, 0, 0],
                    bias=False,
                ),
                norm_c=norm_model(64),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in=32):
                with torch.no_grad():
                    if input_tensor.shape[1] != 32:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue

                    output_tensor = model(input_tensor)
                    output_tensor_gt = model_gt(input_tensor)
                self.assertEqual(
                    output_tensor.shape,
                    output_tensor_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        output_tensor.shape, output_tensor_gt.shape
                    ),
                )
                self.assertTrue(
                    np.allclose(output_tensor.numpy(), output_tensor_gt.numpy())
                )

    @staticmethod
    def _get_inputs(dim_in: int = 3) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random segmentation as test cases.
        shapes = (
            # Forward succeeded.
            (1, dim_in, 3, 7, 7),
            (1, dim_in, 5, 7, 7),
            (1, dim_in, 7, 7, 7),
            (2, dim_in, 3, 7, 7),
            (4, dim_in, 3, 7, 7),
            (8, dim_in, 3, 7, 7),
            (2, dim_in, 3, 7, 14),
            (2, dim_in, 3, 14, 7),
            (2, dim_in, 3, 14, 14),
            # Forward failed.
            (8, dim_in * 2, 3, 7, 7),
            (8, dim_in * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)


class TestResBottleneckBlock(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_res_block(self):
        """
        Test simple ResBlock with different inputs.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = ResBlock(
                branch1_conv=nn.Conv3d(
                    dim_in, dim_out, kernel_size=(1, 1, 1), stride=(1, 1, 1)
                )
                if dim_in != dim_out
                else None,
                branch1_norm=nn.BatchNorm3d(num_features=dim_out)
                if dim_in != dim_out
                else None,
                branch2=BottleneckBlock(
                    conv_a=nn.Conv3d(
                        dim_in,
                        dim_inner,
                        kernel_size=[3, 1, 1],
                        stride=[1, 1, 1],
                        padding=[1, 0, 0],
                        bias=False,
                    ),
                    norm_a=nn.BatchNorm3d(dim_inner),
                    act_a=nn.ReLU(),
                    conv_b=nn.Conv3d(
                        dim_inner,
                        dim_inner,
                        kernel_size=[1, 3, 3],
                        stride=[1, 1, 1],
                        padding=[0, 1, 1],
                        bias=False,
                    ),
                    norm_b=nn.BatchNorm3d(dim_inner),
                    act_b=nn.ReLU(),
                    conv_c=nn.Conv3d(
                        dim_inner,
                        dim_out,
                        kernel_size=[1, 1, 1],
                        stride=[1, 1, 1],
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    norm_c=nn.BatchNorm3d(dim_out),
                ),
                activation=nn.ReLU(),
                branch_fusion=lambda x, y: x + y,
            )

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(input_tensor)
                    continue

                output_tensor = model(input_tensor)

                input_shape = input_tensor.shape
                output_shape = output_tensor.shape
                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    input_shape[2],
                    input_shape[3],
                    input_shape[4],
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_res_block_with_callable(self):
        """
        Test builder `create_res_block` with callable inputs.
        """
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Softmax, nn.Sigmoid, None)
        ):
            model = create_res_block(
                dim_in=32,
                dim_inner=16,
                dim_out=64,
                bottleneck=create_bottleneck_block,
                conv_a_kernel_size=(3, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(1, 0, 0),
                conv_b_kernel_size=(1, 3, 3),
                conv_b_stride=(1, 2, 2),
                conv_b_padding=(0, 1, 1),
                conv_b_num_groups=1,
                conv_b_dilation=(1, 1, 1),
                norm=norm,
                norm_eps=1e-5,
                norm_momentum=0.1,
                activation_bottleneck=activation,
                activation_block=activation,
            )
            model_gt = ResBlock(
                branch1_conv=nn.Conv3d(
                    32, 64, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False
                ),
                branch1_norm=None if norm is None else norm(num_features=64),
                branch2=BottleneckBlock(
                    conv_a=nn.Conv3d(
                        32,
                        16,
                        kernel_size=[3, 1, 1],
                        stride=[1, 1, 1],
                        padding=[1, 0, 0],
                        bias=False,
                    ),
                    norm_a=None if norm is None else norm(16),
                    act_a=None if activation is None else activation(),
                    conv_b=nn.Conv3d(
                        16,
                        16,
                        kernel_size=[1, 3, 3],
                        stride=[1, 2, 2],
                        padding=[0, 1, 1],
                        bias=False,
                    ),
                    norm_b=None if norm is None else norm(16),
                    act_b=None if activation is None else activation(),
                    conv_c=nn.Conv3d(
                        16,
                        64,
                        kernel_size=[1, 1, 1],
                        stride=[1, 1, 1],
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    norm_c=None if norm is None else norm(64),
                ),
                activation=None if activation is None else activation(),
                branch_fusion=lambda x, y: x + y,
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestBottleneckBlock._get_inputs(dim_in=32):
                with torch.no_grad():
                    if input_tensor.shape[1] != 32:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue

                    output_tensor = model(input_tensor)
                    output_tensor_gt = model_gt(input_tensor)

                self.assertEqual(
                    output_tensor.shape,
                    output_tensor_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        output_tensor.shape, output_tensor_gt.shape
                    ),
                )
                self.assertTrue(
                    np.allclose(output_tensor.numpy(), output_tensor_gt.numpy())
                )

    @staticmethod
    def _get_inputs(dim_in: int = 3) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = (
            # Forward succeeded.
            (1, dim_in, 3, 7, 7),
            (1, dim_in, 5, 7, 7),
            (1, dim_in, 7, 7, 7),
            (2, dim_in, 3, 7, 7),
            (4, dim_in, 3, 7, 7),
            (8, dim_in, 3, 7, 7),
            (2, dim_in, 3, 7, 14),
            (2, dim_in, 3, 14, 7),
            (2, dim_in, 3, 14, 14),
            # Forward failed.
            (8, dim_in * 2, 3, 7, 7),
            (8, dim_in * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)


class TestResStageTransform(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_res_stage(self):
        """
        Test simple ResStage with different inputs.
        """
        for dim_in, dim_inner, dim_out in itertools.product(
            (4, 8, 16), (2, 4), (4, 8, 16)
        ):
            model = ResStage(
                res_blocks=nn.ModuleList(
                    [
                        ResBlock(
                            branch1_conv=nn.Conv3d(
                                dim_in, dim_out, kernel_size=(1, 1, 1)
                            )
                            if dim_in != dim_out
                            else None,
                            branch1_norm=nn.BatchNorm3d(num_features=dim_out)
                            if dim_in != dim_out
                            else None,
                            branch2=BottleneckBlock(
                                conv_a=nn.Conv3d(
                                    dim_in,
                                    dim_inner,
                                    kernel_size=[3, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[1, 0, 0],
                                    bias=False,
                                ),
                                norm_a=nn.BatchNorm3d(dim_inner),
                                act_a=nn.ReLU(),
                                conv_b=nn.Conv3d(
                                    dim_inner,
                                    dim_inner,
                                    kernel_size=[1, 3, 3],
                                    stride=[1, 1, 1],
                                    padding=[0, 1, 1],
                                    bias=False,
                                ),
                                norm_b=nn.BatchNorm3d(dim_inner),
                                act_b=nn.ReLU(),
                                conv_c=nn.Conv3d(
                                    dim_inner,
                                    dim_out,
                                    kernel_size=[1, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[0, 0, 0],
                                    bias=False,
                                ),
                                norm_c=nn.BatchNorm3d(dim_out),
                            ),
                            activation=nn.ReLU(),
                            branch_fusion=lambda x, y: x + y,
                        ),
                        ResBlock(
                            branch1_conv=None,
                            branch1_norm=None,
                            branch2=BottleneckBlock(
                                conv_a=nn.Conv3d(
                                    dim_out,
                                    dim_inner,
                                    kernel_size=[3, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[1, 0, 0],
                                    bias=False,
                                ),
                                norm_a=nn.BatchNorm3d(dim_inner),
                                act_a=nn.ReLU(),
                                conv_b=nn.Conv3d(
                                    dim_inner,
                                    dim_inner,
                                    kernel_size=[1, 3, 3],
                                    stride=[1, 1, 1],
                                    padding=[0, 1, 1],
                                    bias=False,
                                ),
                                norm_b=nn.BatchNorm3d(dim_inner),
                                act_b=nn.ReLU(),
                                conv_c=nn.Conv3d(
                                    dim_inner,
                                    dim_out,
                                    kernel_size=[1, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[0, 0, 0],
                                    bias=False,
                                ),
                                norm_c=nn.BatchNorm3d(dim_out),
                            ),
                            activation=nn.ReLU(),
                            branch_fusion=lambda x, y: x + y,
                        ),
                    ]
                )
            )

            # Test forwarding.
            for tensor in TestResStageTransform._get_inputs(dim_in):
                if tensor.shape[1] != dim_in:
                    with self.assertRaises(RuntimeError):
                        out = model(tensor)
                    continue

                out = model(tensor)

                input_shape = tensor.shape
                output_shape = out.shape
                output_shape_gt = (
                    input_shape[0],
                    dim_out,
                    input_shape[2],
                    input_shape[3],
                    input_shape[4],
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_res_stage_with_callable(self):
        """
        Test builder `create_res_stage` with callable inputs.
        """
        dim_in, dim_inner, dim_out = 32, 16, 64
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            model = create_res_stage(
                depth=2,
                dim_in=dim_in,
                dim_inner=dim_inner,
                dim_out=dim_out,
                bottleneck=create_bottleneck_block,
                conv_a_kernel_size=(3, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(1, 0, 0),
                conv_b_kernel_size=(1, 3, 3),
                conv_b_stride=(1, 1, 1),
                conv_b_padding=(0, 1, 1),
                conv_b_num_groups=1,
                conv_b_dilation=(1, 1, 1),
                norm=norm,
                norm_eps=1e-5,
                norm_momentum=0.1,
                activation=activation,
            )
            model_gt = ResStage(
                res_blocks=nn.ModuleList(
                    [
                        ResBlock(
                            branch1_conv=nn.Conv3d(
                                dim_in, dim_out, kernel_size=(1, 1, 1), bias=False
                            )
                            if dim_in != dim_out
                            else None,
                            branch1_norm=None
                            if norm is None
                            else norm(num_features=dim_out)
                            if dim_in != dim_out
                            else None,
                            branch2=BottleneckBlock(
                                conv_a=nn.Conv3d(
                                    dim_in,
                                    dim_inner,
                                    kernel_size=[3, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[1, 0, 0],
                                    bias=False,
                                ),
                                norm_a=None if norm is None else norm(dim_inner),
                                act_a=None if activation is None else activation(),
                                conv_b=nn.Conv3d(
                                    dim_inner,
                                    dim_inner,
                                    kernel_size=[1, 3, 3],
                                    stride=[1, 1, 1],
                                    padding=[0, 1, 1],
                                    bias=False,
                                ),
                                norm_b=None if norm is None else norm(dim_inner),
                                act_b=None if activation is None else activation(),
                                conv_c=nn.Conv3d(
                                    dim_inner,
                                    dim_out,
                                    kernel_size=[1, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[0, 0, 0],
                                    bias=False,
                                ),
                                norm_c=None if norm is None else norm(dim_out),
                            ),
                            activation=None if activation is None else activation(),
                            branch_fusion=lambda x, y: x + y,
                        ),
                        ResBlock(
                            branch1_conv=None,
                            branch1_norm=None,
                            branch2=BottleneckBlock(
                                conv_a=nn.Conv3d(
                                    dim_out,
                                    dim_inner,
                                    kernel_size=[3, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[1, 0, 0],
                                    bias=False,
                                ),
                                norm_a=None if norm is None else norm(dim_inner),
                                act_a=None if activation is None else activation(),
                                conv_b=nn.Conv3d(
                                    dim_inner,
                                    dim_inner,
                                    kernel_size=[1, 3, 3],
                                    stride=[1, 1, 1],
                                    padding=[0, 1, 1],
                                    bias=False,
                                ),
                                norm_b=None if norm is None else norm(dim_inner),
                                act_b=None if activation is None else activation(),
                                conv_c=nn.Conv3d(
                                    dim_inner,
                                    dim_out,
                                    kernel_size=[1, 1, 1],
                                    stride=[1, 1, 1],
                                    padding=[0, 0, 0],
                                    bias=False,
                                ),
                                norm_c=None if norm is None else norm(dim_out),
                            ),
                            activation=None if activation is None else activation(),
                            branch_fusion=lambda x, y: x + y,
                        ),
                    ]
                )
            )
            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for tensor in TestResStageTransform._get_inputs(dim_in=dim_in):
                with torch.no_grad():
                    if tensor.shape[1] != 32:
                        with self.assertRaises(RuntimeError):
                            out = model(tensor)
                        continue

                    out = model(tensor)
                    out_gt = model_gt(tensor)

                self.assertEqual(
                    out.shape,
                    out_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        out.shape, out_gt.shape
                    ),
                )
                self.assertTrue(np.allclose(out.numpy(), out_gt.numpy()))

    @staticmethod
    def _get_inputs(dim_in: int = 3) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = (
            # Forward succeeded.
            (1, dim_in, 3, 7, 7),
            (1, dim_in, 5, 7, 7),
            (1, dim_in, 7, 7, 7),
            (2, dim_in, 3, 7, 7),
            (4, dim_in, 3, 7, 7),
            (8, dim_in, 3, 7, 7),
            (2, dim_in, 3, 7, 14),
            (2, dim_in, 3, 14, 7),
            (2, dim_in, 3, 14, 14),
            # Forward failed.
            (8, dim_in * 2, 3, 7, 7),
            (8, dim_in * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)


class TestResNet(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def _build_resnet(
        self,
        input_channel,
        input_clip_length,
        input_crop_size,
        model_depth,
        norm,
        activation,
    ):
        _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
        stem_dim_out = 8
        model_num_class = 10
        stages = []
        # create the Stem for ResNet
        stem = ResNetBasicStem(
            conv=nn.Conv3d(
                input_channel,
                stem_dim_out,
                kernel_size=[3, 7, 7],
                stride=[1, 2, 2],
                padding=[1, 3, 3],
                bias=False,
            ),
            norm=None if norm is None else norm(stem_dim_out),
            activation=None if activation is None else activation(),
            pool=nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            ),
        )
        stages.append(stem)

        # get the number of Blocks for each Stage
        stage_depths = _MODEL_STAGE_DEPTH[model_depth]

        stage_dim_in = stem_dim_out
        stage_dim_out = stage_dim_in * 4
        stage_spatial_stride = (2, 1, 1, 1)
        stage_temporal_stride = (2, 1, 1, 1)

        # create each Stage for ResNet
        for i in range(len(stage_depths)):
            stage_dim_inner = stage_dim_out // 4
            depth = stage_depths[i]

            block_dim_in = stage_dim_in
            block_dim_inner = stage_dim_inner
            block_dim_out = stage_dim_out

            blocks = []
            for j in range(depth):
                spatial_stride = stage_spatial_stride[i] if j == 0 else 1
                temporal_stride = stage_temporal_stride[i] if j == 0 else 1
                # create each Block for the Stage
                block = ResBlock(
                    branch1_conv=nn.Conv3d(
                        block_dim_in,
                        block_dim_out,
                        kernel_size=(1, 1, 1),
                        stride=(temporal_stride, spatial_stride, spatial_stride),
                        bias=False,
                    )
                    if block_dim_in != block_dim_out
                    else None,
                    branch1_norm=None
                    if norm is None
                    else norm(block_dim_out)
                    if block_dim_in != block_dim_out
                    else None,
                    branch2=BottleneckBlock(
                        conv_a=nn.Conv3d(
                            block_dim_in,
                            block_dim_inner,
                            kernel_size=[3, 1, 1],
                            stride=[temporal_stride, 1, 1],
                            padding=[1, 0, 0],
                            bias=False,
                        ),
                        norm_a=None if norm is None else norm(block_dim_inner),
                        act_a=None if activation is None else activation(),
                        conv_b=nn.Conv3d(
                            block_dim_inner,
                            block_dim_inner,
                            kernel_size=[1, 3, 3],
                            stride=[1, spatial_stride, spatial_stride],
                            padding=[0, 1, 1],
                            bias=False,
                        ),
                        norm_b=None if norm is None else norm(block_dim_inner),
                        act_b=None if activation is None else activation(),
                        conv_c=nn.Conv3d(
                            block_dim_inner,
                            block_dim_out,
                            kernel_size=[1, 1, 1],
                            stride=[1, 1, 1],
                            padding=[0, 0, 0],
                            bias=False,
                        ),
                        norm_c=None if norm is None else norm(block_dim_out),
                    ),
                    activation=None if activation is None else activation(),
                    branch_fusion=lambda x, y: x + y,
                )

                block_dim_in = block_dim_out
                blocks.append(block)

            stage = ResStage(nn.ModuleList(blocks))
            stages.append(stage)

            stage_dim_in = stage_dim_out
            stage_dim_out = stage_dim_out * 2

        # Create Head for ResNet
        total_spatial_stride = 4 * np.prod(stage_spatial_stride)
        total_temporal_stride = np.prod(stage_temporal_stride)
        head_pool_kernel_size = (
            input_clip_length // total_temporal_stride,
            input_crop_size // total_spatial_stride,
            input_crop_size // total_spatial_stride,
        )

        head = ResNetBasicHead(
            proj=nn.Linear(stage_dim_in, model_num_class),
            activation=nn.Softmax(),
            pool=nn.AvgPool3d(kernel_size=head_pool_kernel_size, stride=[1, 1, 1]),
            dropout=None,
            output_pool=nn.AdaptiveAvgPool3d(1),
        )
        stages.append(head)

        return (Net(blocks=nn.ModuleList(stages)), model_num_class)

    def test_create_resnet(self):
        """
        Test simple ResNet with different inputs.
        """
        for input_channel, input_clip_length, input_crop_size in itertools.product(
            (3, 2), (2, 4), (56, 64)
        ):
            model_depth = 50
            model, num_class = self._build_resnet(
                input_channel,
                input_clip_length,
                input_crop_size,
                model_depth,
                nn.BatchNorm3d,
                nn.ReLU,
            )

            # Test forwarding.
            for tensor in TestResNet._get_inputs(
                input_channel, input_clip_length, input_crop_size
            ):
                if tensor.shape[1] != input_channel:
                    with self.assertRaises(RuntimeError):
                        out = model(tensor)
                    continue

                out = model(tensor)

                output_shape = out.shape
                output_shape_gt = (tensor.shape[0], num_class)

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_resnet_with_callable(self):
        """
        Test builder `create_resnet` with callable inputs.
        """
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            input_channel = 3
            input_clip_length = 4
            input_crop_size = 56
            model_depth = 50
            stage_spatial_stride = (2, 1, 1, 1)
            stage_temporal_stride = (2, 1, 1, 1)
            model_gt, num_class = self._build_resnet(
                input_channel,
                input_clip_length,
                input_crop_size,
                model_depth,
                norm,
                activation,
            )

            total_spatial_stride = 4 * np.prod(stage_spatial_stride)
            total_temporal_stride = np.prod(stage_temporal_stride)
            head_pool_kernel_size = (
                input_clip_length // total_temporal_stride,
                input_crop_size // total_spatial_stride,
                input_crop_size // total_spatial_stride,
            )

            model = create_resnet(
                input_channel=input_channel,
                model_depth=50,
                model_num_class=num_class,
                dropout_rate=0,
                norm=norm,
                activation=activation,
                stem_dim_out=8,
                stem_conv_kernel_size=(3, 7, 7),
                stem_conv_stride=(1, 2, 2),
                stem_pool=nn.MaxPool3d,
                stem_pool_kernel_size=(1, 3, 3),
                stem_pool_stride=(1, 2, 2),
                stage_conv_a_kernel_size=((3, 1, 1),) * 4,
                stage_conv_b_kernel_size=((1, 3, 3),) * 4,
                stage_spatial_stride=stage_spatial_stride,
                stage_temporal_stride=stage_temporal_stride,
                bottleneck=create_bottleneck_block,
                head_pool=nn.AvgPool3d,
                head_pool_kernel_size=head_pool_kernel_size,
                head_output_size=(1, 1, 1),
                head_activation=nn.Softmax,
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for tensor in TestResNet._get_inputs(
                input_channel, input_clip_length, input_crop_size
            ):
                with torch.no_grad():
                    if tensor.shape[1] != input_channel:
                        with self.assertRaises(RuntimeError):
                            out = model(tensor)
                        continue

                    out = model(tensor)
                    out_gt = model_gt(tensor)

                self.assertEqual(
                    out.shape,
                    out_gt.shape,
                    "Output shape {} is different from expected shape {}".format(
                        out.shape, out_gt.shape
                    ),
                )
                self.assertTrue(
                    np.allclose(out.numpy(), out_gt.numpy(), rtol=1e-1, atol=1e-1)
                )

    def test_create_acoustic_resnet_with_callable(self):
        """
        Test builder `create_acoustic_resnet` with callable inputs.
        """
        _input_channel = 1
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            model = create_acoustic_resnet(
                input_channel=_input_channel,
                stem_conv_kernel_size=(3, 3, 3),
                stem_conv_padding=(1, 1, 1),
                model_depth=50,
                model_num_class=400,
                dropout_rate=0,
                norm=norm,
                activation=activation,
                stem_dim_out=8,
                stem_pool=nn.MaxPool3d,
                stem_pool_kernel_size=(1, 3, 3),
                stem_pool_stride=(1, 2, 2),
                stage_conv_a_kernel_size=(3, 1, 1),
                stage_conv_b_kernel_size=(1, 3, 3),
                stage_spatial_stride=(2, 1, 1, 1),
                stage_temporal_stride=(2, 1, 1, 1),
                head_pool=nn.AvgPool3d,
                head_output_size=(1, 1, 1),
                head_activation=nn.Softmax,
            )

            # Test forwarding.
            for tensor in TestResNet._get_inputs(_input_channel, 1, 56):
                with torch.no_grad():
                    if tensor.shape[1] != _input_channel:
                        with self.assertRaises(RuntimeError):
                            model(tensor)
                        continue
                    model(tensor)

    def test_load_hubconf(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
        )
        input_channel = 3
        input_clip_length = 2
        input_crop_size = 56
        model = torch.hub.load(
            repo_or_dir=path, source="local", model="slow_r50", pretrained=False
        )
        self.assertIsNotNone(model)

        # Test forwarding.
        for tensor in TestResNet._get_inputs(
            input_channel, input_clip_length, input_crop_size
        ):
            with torch.no_grad():
                if tensor.shape[1] != input_channel:
                    with self.assertRaises(RuntimeError):
                        model(tensor)
                    continue

    @staticmethod
    def _get_inputs(
        channel: int = 3, clip_length: int = 8, crop_size: int = 224
    ) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = (
            (1, channel, clip_length, crop_size, crop_size),
            (2, channel, clip_length, crop_size, crop_size),
        )
        for shape in shapes:
            yield torch.rand(shape)

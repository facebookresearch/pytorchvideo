import itertools
import unittest

import numpy as np
import torch
from pytorchvideo.models.resnet import (
    BottleneckBlock,
    ResBlock,
    ResStage,
    create_default_bottleneck_block,
    create_default_res_block,
    create_default_res_stage,
)
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

    def test_create_default_bottleneck_block_with_callable(self):
        """
        Test default builder `create_default_bottleneck_block` with callable inputs.
        """
        for (norm_model, act_model) in itertools.product(
            (nn.BatchNorm3d,), (nn.ReLU, nn.Softmax, nn.Sigmoid)
        ):
            model = create_default_bottleneck_block(
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

    def test_create_default_res_block(self):
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

    def test_create_default_res_block_with_callable(self):
        """
        Test default builder `create_default_res_block` with callable inputs.
        """
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Softmax, nn.Sigmoid, None)
        ):
            model = create_default_res_block(
                dim_in=32,
                dim_inner=16,
                dim_out=64,
                bottleneck=create_default_bottleneck_block,
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
                activation=activation,
            )
            model_gt = ResBlock(
                branch1_conv=nn.Conv3d(32, 64, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
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
                res_blocks=[
                    ResBlock(
                        branch1_conv=nn.Conv3d(dim_in, dim_out, kernel_size=(1, 1, 1))
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
                    ),
                ]
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

    def test_create_default_res_stage_with_callable(self):
        """
        Test default builder `create_default_res_stage` with callable inputs.
        """
        dim_in, dim_inner, dim_out = 32, 16, 64
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            model = create_default_res_stage(
                depth=2,
                dim_in=dim_in,
                dim_inner=dim_inner,
                dim_out=dim_out,
                bottleneck=create_default_bottleneck_block,
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
                res_blocks=[
                    ResBlock(
                        branch1_conv=nn.Conv3d(dim_in, dim_out, kernel_size=(1, 1, 1))
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
                    ),
                ]
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

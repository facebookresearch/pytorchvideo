# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest

import numpy as np
import torch
from pytorchvideo.layers.convolutions import ConvReduce3D
from pytorchvideo.models.stem import (
    ResNetBasicStem,
    create_acoustic_res_basic_stem,
    create_res_basic_stem,
)
from torch import nn


class TestResNetBasicStem(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_simple_stem(self):
        """
        Test simple ResNetBasicStem (without pooling layer).
        """
        for input_dim, output_dim in itertools.product((2, 3), (4, 8, 16)):
            model = ResNetBasicStem(
                conv=nn.Conv3d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm=nn.BatchNorm3d(output_dim),
                activation=nn.ReLU(),
                pool=None,
            )

            # Test forwarding.
            for tensor in TestResNetBasicStem._get_inputs(input_dim):
                if tensor.shape[1] != input_dim:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(tensor)
                    continue
                else:
                    output_tensor = model(tensor)

                input_shape = tensor.shape
                output_shape = output_tensor.shape
                output_shape_gt = (
                    input_shape[0],
                    output_dim,
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

    def test_create_stem_with_conv_reduced_3d(self):
        """
        Test simple ResNetBasicStem with ConvReduce3D.
        """
        for input_dim, output_dim in itertools.product((2, 3), (4, 8, 16)):
            model = ResNetBasicStem(
                conv=ConvReduce3D(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=(False, False),
                ),
                norm=nn.BatchNorm3d(output_dim),
                activation=nn.ReLU(),
                pool=None,
            )

            # Test forwarding.
            for tensor in TestResNetBasicStem._get_inputs(input_dim):
                if tensor.shape[1] != input_dim:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(tensor)
                    continue
                else:
                    output_tensor = model(tensor)

                input_shape = tensor.shape
                output_shape = output_tensor.shape
                output_shape_gt = (
                    input_shape[0],
                    output_dim,
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

    def test_create_complex_stem(self):
        """
        Test complex ResNetBasicStem.
        """
        for input_dim, output_dim in itertools.product((2, 3), (4, 8, 16)):
            model = ResNetBasicStem(
                conv=nn.Conv3d(
                    input_dim,
                    output_dim,
                    kernel_size=[3, 7, 7],
                    stride=[1, 2, 2],
                    padding=[1, 3, 3],
                    bias=False,
                ),
                norm=nn.BatchNorm3d(output_dim),
                activation=nn.ReLU(),
                pool=nn.MaxPool3d(
                    kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
                ),
            )

            # Test forwarding.
            for input_tensor in TestResNetBasicStem._get_inputs(input_dim):
                if input_tensor.shape[1] != input_dim:
                    with self.assertRaises(Exception):
                        output_tensor = model(input_tensor)
                    continue
                else:
                    output_tensor = model(input_tensor)

                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                output_shape_gt = (
                    input_shape[0],
                    output_dim,
                    input_shape[2],
                    (((input_shape[3] - 1) // 2 + 1) - 1) // 2 + 1,
                    (((input_shape[4] - 1) // 2 + 1) - 1) // 2 + 1,
                )

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    def test_create_stem_with_callable(self):
        """
        Test builder `create_res_basic_stem` with callable inputs.
        """
        for (pool, activation, norm) in itertools.product(
            (nn.AvgPool3d, nn.MaxPool3d, None),
            (nn.ReLU, nn.Softmax, nn.Sigmoid, None),
            (nn.BatchNorm3d, None),
        ):
            model = create_res_basic_stem(
                in_channels=3,
                out_channels=64,
                pool=pool,
                activation=activation,
                norm=norm,
            )
            model_gt = ResNetBasicStem(
                conv=nn.Conv3d(
                    3,
                    64,
                    kernel_size=[3, 7, 7],
                    stride=[1, 2, 2],
                    padding=[1, 3, 3],
                    bias=False,
                ),
                norm=None if norm is None else norm(64),
                activation=None if activation is None else activation(),
                pool=None
                if pool is None
                else pool(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestResNetBasicStem._get_inputs():
                with torch.no_grad():
                    if input_tensor.shape[1] != 3:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue
                    else:
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

    def test_create_acoustic_stem_with_callable(self):
        """
        Test builder `create_acoustic_res_basic_stem` with callable
        inputs.
        """
        for (pool, activation, norm) in itertools.product(
            (nn.AvgPool3d, nn.MaxPool3d, None),
            (nn.ReLU, nn.Softmax, nn.Sigmoid, None),
            (nn.BatchNorm3d, None),
        ):
            model = create_acoustic_res_basic_stem(
                in_channels=3,
                out_channels=64,
                pool=pool,
                activation=activation,
                norm=norm,
            )
            model_gt = ResNetBasicStem(
                conv=ConvReduce3D(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=((3, 1, 1), (1, 7, 7)),
                    stride=((1, 1, 1), (1, 1, 1)),
                    padding=((1, 0, 0), (0, 3, 3)),
                    bias=(False, False),
                ),
                norm=None if norm is None else norm(64),
                activation=None if activation is None else activation(),
                pool=None
                if pool is None
                else pool(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
            )

            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestResNetBasicStem._get_inputs():
                with torch.no_grad():
                    if input_tensor.shape[1] != 3:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue
                    else:
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
    def _get_inputs(input_dim: int = 3) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        shapes = (
            # Forward succeeded.
            (1, input_dim, 3, 7, 7),
            (1, input_dim, 5, 7, 7),
            (1, input_dim, 7, 7, 7),
            (2, input_dim, 3, 7, 7),
            (4, input_dim, 3, 7, 7),
            (8, input_dim, 3, 7, 7),
            (2, input_dim, 3, 7, 14),
            (2, input_dim, 3, 14, 7),
            (2, input_dim, 3, 14, 14),
            # Forward failed.
            (8, input_dim * 2, 3, 7, 7),
            (8, input_dim * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)

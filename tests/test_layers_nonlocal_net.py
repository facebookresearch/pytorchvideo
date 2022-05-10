# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest
from typing import Iterable

import numpy as np
import torch
from pytorchvideo.layers.nonlocal_net import NonLocal, create_nonlocal
from torch import nn


class TestNonlocal(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_build_nonlocal(self):
        """
        Test Nonlocal model builder.
        """
        for dim_in, dim_inner, pool, norm, instantiation in itertools.product(
            (4, 8),
            (2, 4),
            (None, nn.MaxPool3d(2)),
            (None, nn.BatchNorm3d),
            ("dot_product", "softmax"),
        ):
            model = NonLocal(
                conv_theta=nn.Conv3d(
                    dim_in, dim_inner, kernel_size=1, stride=1, padding=0
                ),
                conv_phi=nn.Conv3d(
                    dim_in, dim_inner, kernel_size=1, stride=1, padding=0
                ),
                conv_g=nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0),
                conv_out=nn.Conv3d(
                    dim_inner, dim_in, kernel_size=1, stride=1, padding=0
                ),
                pool=pool,
                norm=norm(dim_in) if norm is not None else None,
                instantiation=instantiation,
            )

            # Test forwarding.
            for input_tensor in TestNonlocal._get_inputs(input_dim=dim_in):
                if input_tensor.shape[1] != dim_in:
                    with self.assertRaises(RuntimeError):
                        output_tensor = model(input_tensor)
                    continue
                else:
                    output_tensor = model(input_tensor)

                input_shape = input_tensor.shape
                output_shape = output_tensor.shape

                self.assertEqual(
                    input_shape,
                    output_shape,
                    "Input shape {} is different from output shape {}".format(
                        input_shape, output_shape
                    ),
                )

    def test_nonlocal_builder(self):
        """
        Test builder `create_nonlocal`.
        """
        for dim_in, dim_inner, pool_size, norm, instantiation in itertools.product(
            (4, 8),
            (2, 4),
            ((1, 1, 1), (2, 2, 2)),
            (None, nn.BatchNorm3d),
            ("dot_product", "softmax"),
        ):
            conv_theta = nn.Conv3d(
                dim_in, dim_inner, kernel_size=1, stride=1, padding=0
            )
            conv_phi = nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0)
            conv_g = nn.Conv3d(dim_in, dim_inner, kernel_size=1, stride=1, padding=0)
            conv_out = nn.Conv3d(dim_inner, dim_in, kernel_size=1, stride=1, padding=0)
            if norm is None:
                norm_model = None
            else:
                norm_model = norm(num_features=dim_in)
            if isinstance(pool_size, Iterable) and any(size > 1 for size in pool_size):
                pool_model = nn.MaxPool3d(
                    kernel_size=pool_size, stride=pool_size, padding=[0, 0, 0]
                )
            else:
                pool_model = None

            model = create_nonlocal(
                dim_in=dim_in,
                dim_inner=dim_inner,
                pool_size=pool_size,
                instantiation=instantiation,
                norm=norm,
            )

            model_gt = NonLocal(
                conv_theta=conv_theta,
                conv_phi=conv_phi,
                conv_g=conv_g,
                conv_out=conv_out,
                pool=pool_model,
                norm=norm_model,
                instantiation=instantiation,
            )
            model.load_state_dict(
                model_gt.state_dict(), strict=True
            )  # explicitly use strict mode.

            # Test forwarding.
            for input_tensor in TestNonlocal._get_inputs(input_dim=dim_in):
                with torch.no_grad():
                    if input_tensor.shape[1] != dim_in:
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
    def _get_inputs(input_dim: int = 8) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        shapes = (
            # Forward succeeded.
            (1, input_dim, 5, 7, 7),
            (2, input_dim, 5, 7, 7),
            (4, input_dim, 5, 7, 7),
            (4, input_dim, 5, 7, 7),
            (4, input_dim, 7, 7, 7),
            (4, input_dim, 7, 7, 14),
            (4, input_dim, 7, 14, 7),
            (4, input_dim, 7, 14, 14),
            # Forward failed.
            (8, input_dim * 2, 3, 7, 7),
            (8, input_dim * 4, 5, 7, 7),
        )
        for shape in shapes:
            yield torch.rand(shape)

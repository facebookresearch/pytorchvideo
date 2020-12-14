# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.layers.swish import Swish
from pytorchvideo.models.x3d import create_x3d, create_x3d_bottleneck_block
from torch import nn


class TestX3d(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_x3d(self):
        """
        Test different version of X3D: X3D-XS, X3D-S, X3D-M, X3D-L.
        """
        for input_clip_length, input_crop_size, depth_factor in [
            (4, 160, 2.2),
            (13, 160, 2.2),
            (16, 224, 2.2),
            (16, 312, 5.0),
        ]:
            model = create_x3d(
                input_clip_length=input_clip_length,
                input_crop_size=input_crop_size,
                model_num_class=400,
                dropout_rate=0.5,
                width_factor=2.0,
                depth_factor=depth_factor,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                stem_dim_in=12,
                stem_conv_kernel_size=(5, 3, 3),
                stem_conv_stride=(1, 2, 2),
                stage_conv_kernel_size=((3, 3, 3),) * 4,
                stage_spatial_stride=(2, 2, 2, 2),
                stage_temporal_stride=(1, 1, 1, 1),
                bottleneck=create_x3d_bottleneck_block,
                bottleneck_factor=2.25,
                se_ratio=0.0625,
                inner_act=Swish,
                head_dim_out=2048,
                head_pool_act=nn.ReLU,
                head_bn_lin5_on=False,
                head_activation=nn.Softmax,
            )

            # Test forwarding.
            for tensor in TestX3d._get_inputs(input_clip_length, input_crop_size):
                if tensor.shape[1] != 3:
                    with self.assertRaises(RuntimeError):
                        out = model(tensor)
                    continue

                out = model(tensor)

                output_shape = out.shape
                output_shape_gt = (tensor.shape[0], 400)

                self.assertEqual(
                    output_shape,
                    output_shape_gt,
                    "Output shape {} is different from expected shape {}".format(
                        output_shape, output_shape_gt
                    ),
                )

    @staticmethod
    def _get_inputs(clip_length: int = 4, crop_size: int = 160) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = (
            (1, 3, clip_length, crop_size, crop_size),
            (2, 3, clip_length, crop_size, crop_size),
        )
        for shape in shapes:
            yield torch.rand(shape)

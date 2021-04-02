# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
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
        To test different versions of X3D, set the input to:
        X3D-XS: (4, 160, 2.0, 2.2, 2.25)
        X3D-S: (13, 160, 2.0, 2.2, 2.25)
        X3D-M: (16, 224, 2.0, 2.2, 2.25)
        X3D-L: (16, 312, 2.0, 5.0, 2.25)

        Each of the parameters corresponds to input_clip_length, input_crop_size,
        width_factor, depth_factor and bottleneck_factor.
        """
        for (
            input_clip_length,
            input_crop_size,
            width_factor,
            depth_factor,
            bottleneck_factor,
        ) in [
            (4, 160, 2.0, 2.2, 2.25),
        ]:
            model = create_x3d(
                input_clip_length=input_clip_length,
                input_crop_size=input_crop_size,
                model_num_class=400,
                dropout_rate=0.5,
                width_factor=width_factor,
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
                bottleneck_factor=bottleneck_factor,
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

    def test_load_hubconf(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
        )
        for (input_clip_length, input_crop_size, model_name) in [
            (4, 160, "x3d_xs"),
            (13, 160, "x3d_s"),
            (16, 224, "x3d_m"),
        ]:
            model = torch.hub.load(
                repo_or_dir=path,
                source="local",
                model=model_name,
                pretrained=False,
                head_output_with_global_average=True,
            )
            self.assertIsNotNone(model)

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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import unittest

import torch
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import create_x3d


class TestEfficientX3d(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_x3d(self):
        """
        To test different versions, set the (expansion, clip_length, crop_size) to:
        X3D-XS: ("XS", 4, 160)
        X3D-S: ("S", 13, 160)
        X3D-M: ("M", 16, 224)
        X3D-L: ("L", 16, 312)
        """
        for (expansion, input_clip_length, input_crop_size,) in [
            ("XS", 4, 160),
        ]:
            model = create_x3d(expansion=expansion)

            # Test forwarding.
            for tensor in TestEfficientX3d._get_inputs(
                input_clip_length, input_crop_size
            ):
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
            (4, 160, "efficient_x3d_xs"),
            (13, 160, "efficient_x3d_s"),
        ]:
            model = torch.hub.load(
                repo_or_dir=path,
                source="local",
                model=model_name,
                pretrained=False,
            )
            self.assertIsNotNone(model)

            # Test forwarding.
            for tensor in TestEfficientX3d._get_inputs(
                input_clip_length, input_crop_size
            ):
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

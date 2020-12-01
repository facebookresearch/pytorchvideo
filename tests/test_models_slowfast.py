# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest
from typing import Tuple

import torch
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.transforms.functional import repeat_temporal_frames_subsample
from torch import nn


class TestSlowFast(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_slowfast_with_callable(self):
        """
        Test builder `create_slowfast` with callable inputs.
        """
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            input_clip_length = 32
            input_crop_size = 224
            input_channel = 3

            model = create_slowfast(
                slowfast_channel_reduction_ratio=8,
                slowfast_conv_channel_fusion_ratio=2,
                slowfast_fusion_conv_kernel_size=(7, 1, 1),
                slowfast_fusion_conv_stride=(4, 1, 1),
                input_channels=(input_channel,) * 2,
                model_depth=18,
                model_num_class=400,
                dropout_rate=0,
                norm=norm,
                activation=activation,
            )

            # Test forwarding.
            for tensor in TestSlowFast._get_inputs(
                input_channel, input_clip_length, input_crop_size
            ):
                with torch.no_grad():
                    if tensor[0].shape[1] != input_channel:
                        with self.assertRaises(RuntimeError):
                            model(tensor)
                        continue

                    model(tensor)

    @staticmethod
    def _get_inputs(
        channel: int = 3,
        clip_length: int = 8,
        crop_size: int = 224,
        frame_ratios: Tuple[int] = (4, 1),
    ) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = ((1, channel, clip_length, crop_size, crop_size),)
        for shape in shapes:
            yield repeat_temporal_frames_subsample(
                torch.rand(shape), frame_ratios=frame_ratios, temporal_dim=2
            )

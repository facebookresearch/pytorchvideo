# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest
from typing import Tuple

import torch
from pytorchvideo.models.audio_visual_slowfast import create_audio_visual_slowfast
from pytorchvideo.transforms.functional import uniform_temporal_subsample_repeated
from torch import nn


class TestAVSlowFast(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_avslowfast_with_callable(self):
        """
        Test builder `create_audio_visual_slowfast` with callable inputs.
        """
        for (norm, activation) in itertools.product(
            (nn.BatchNorm3d, None), (nn.ReLU, nn.Sigmoid, None)
        ):
            input_channel = 3

            model = create_audio_visual_slowfast(
                input_channels=(input_channel, input_channel, 1),
                model_depth=18,
                norm=norm,
                activation=activation,
            )

            # Test forwarding.
            for tensor in TestAVSlowFast._get_inputs(input_channel):
                with torch.no_grad():
                    if tensor[0].shape[1] != input_channel:
                        with self.assertRaises(RuntimeError):
                            model(tensor)
                        continue

                    model(tensor)

    @staticmethod
    def _get_inputs(
        channel: int = 3,
        clip_length: int = 64,
        audio_clip_length: int = 128,
        crop_size: int = 224,
        audio_size: int = 80,
        frame_ratios: Tuple[int] = (8, 2),
        audio_frame_ratio: int = 1,
    ) -> Tuple[torch.Tensor]:
        """
        Provide different tensors as test cases.

        Yield:
            Tuple[torch.Tensor]: tensors as test case input.
        """
        # Prepare random inputs as test cases.
        shape = (1, channel, clip_length, crop_size, crop_size)
        audio_shape = (1, 1, audio_clip_length, 1, audio_size)
        output = uniform_temporal_subsample_repeated(
            torch.rand(shape), frame_ratios=frame_ratios, temporal_dim=2
        )
        yield output + uniform_temporal_subsample_repeated(
            torch.rand(audio_shape), frame_ratios=(audio_frame_ratio,), temporal_dim=2
        )

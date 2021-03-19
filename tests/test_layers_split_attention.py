import itertools
import unittest
from typing import Tuple

import torch
from pytorchvideo.layers.fb.octave_conv import get_dim_high_low
from pytorchvideo.layers.fb.split_attention import OctaveSplitAttention, SplitAttention


class TestSplitAttention(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_build_split_attention(self):
        """
        Test SplitAttention model builder for both 2D and 3D cases.
        """
        for ndim in (2, 3):
            for dim_in, splits, channel_ratio in itertools.product(
                (32, 48), (1, 2, 4), (0.25, 0.5)
            ):
                model = SplitAttention(
                    dim_in,
                    splits,
                    ndim=ndim,
                    channel_ratio=channel_ratio,
                )

                # Test forwarding.
                for input_tensor in TestSplitAttention._get_inputs(
                    ndim=ndim,
                    input_dim=dim_in,
                    splits=splits,
                ):
                    if input_tensor[0].shape[1] != dim_in:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue
                    else:
                        output_tensor = model(input_tensor)

                    input_shape = input_tensor[0].shape
                    output_shape = output_tensor.shape

                    self.assertEqual(
                        input_shape,
                        output_shape,
                        "Input shape {} is different from output shape {}".format(
                            input_shape, output_shape
                        ),
                    )

    def test_build_octave_split_attention(self):
        """
        Test OctaveSplitAttention model builder for both 2D and 3D cases.
        """
        for ndim in (2, 3):
            for dim_in, splits, channel_ratio, octave_ratio in itertools.product(
                (32, 48), (1, 2, 4), (0.25, 0.5), (0.25, 0.5)
            ):
                model = OctaveSplitAttention(
                    dim_in,
                    splits,
                    ndim=ndim,
                    channel_ratio=channel_ratio,
                    octave_ratio=octave_ratio,
                )

                # Test forwarding.
                for input_tensor in TestSplitAttention._get_octave_inputs(
                    ndim=ndim,
                    input_dim=dim_in,
                    splits=splits,
                    octave_ratio=octave_ratio,
                ):
                    if sum(input_tensor[0][i].shape[1] for i in range(2)) != dim_in:
                        with self.assertRaises(RuntimeError):
                            output_tensor = model(input_tensor)
                        continue
                    else:
                        output_tensor = model(input_tensor)

                    # check both high-frequency and low-frequency feature
                    for i in range(2):
                        input_shape = input_tensor[0][i].shape
                        output_shape = output_tensor[i].shape

                        self.assertEqual(
                            input_shape,
                            output_shape,
                            "Input shape {} is different from output shape {}".format(
                                input_shape, output_shape
                            ),
                        )

    @staticmethod
    def _get_inputs(
        ndim: int = 2, input_dim: int = 8, splits: int = 1
    ) -> Tuple[torch.Tensor]:
        """
        Provide different tensors as test cases.

        Yield:
            tensors as test case input.
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
            yield [
                torch.rand(shape if ndim == 3 else shape[:2] + shape[3:])
                for _i in range(splits)
            ]

    @staticmethod
    def _get_octave_inputs(
        ndim: int = 2, input_dim: int = 8, splits: int = 1, octave_ratio: float = 0.5
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Provide different tensors as test cases.

        Yield:
            tensors as test case input.
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
            dim = shape[1]
            dim_high, dim_low = get_dim_high_low(octave_ratio, dim)

            yield [
                [
                    torch.rand(
                        (shape[0], dim_high, shape[2], shape[3], shape[4])
                        if ndim == 3
                        else (shape[0], dim_high, shape[3], shape[4])
                    ),
                    torch.rand(
                        (shape[0], dim_low, shape[2], shape[3], shape[4])
                        if ndim == 3
                        else (shape[0], dim_low, shape[3], shape[4])
                    ),
                ]
                for _i in range(splits)
            ]

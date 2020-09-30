import itertools
import unittest

import torch
from pytorchvideo.models.csn import create_default_csn
from pytorchvideo.models.resnet import create_default_bottleneck_block
from torch import nn


class TestCSN(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_csn(self):
        """
        Test simple CSN with different inputs.
        """
        for input_channel, input_clip_length, input_crop_size in itertools.product(
            (3, 2), (4, 8), (56, 64)
        ):
            model = create_default_csn(
                input_channel=input_channel,
                input_clip_length=input_clip_length,
                input_crop_size=input_crop_size,
                model_depth=50,
                model_num_class=400,
                dropout_rate=0,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                stem_dim_out=8,
                stem_conv_kernel_size=(3, 7, 7),
                stem_conv_stride=(1, 2, 2),
                stage_conv_a_kernel_size=(1, 1, 1),
                stage_conv_b_kernel_size=(3, 3, 3),
                stage_conv_b_width_per_group=1,
                stage_spatial_stride=(1, 2, 2, 2),
                stage_temporal_stride=(1, 2, 2, 1),
                bottleneck=create_default_bottleneck_block,
                head_pool=nn.AvgPool3d,
                head_output_size=(1, 1, 1),
                head_activation=nn.Softmax,
            )

            # Test forwarding.
            for tensor in TestCSN._get_inputs(
                input_channel, input_clip_length, input_crop_size
            ):
                if tensor.shape[1] != input_channel:
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
    def _get_inputs(
        channel: int = 3, clip_length: int = 4, crop_size: int = 112
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
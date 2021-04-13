# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.layers import make_fusion_layer


class TestFusion(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

        self.fake_input_1 = torch.Tensor(
            [[[4, -2], [3, 0]], [[0, 2], [4, 3]], [[3, 1], [5, 2]]]
        ).float()
        self.fake_input_2 = torch.Tensor(
            [[[1, 2], [3, 4]], [[5, 6], [6, 5]], [[4, 3], [2, 1]]]
        ).float()

    def test_reduce_fusion_layers(self):
        expected_output_for_method = {
            "max": torch.Tensor(
                [[[4, 2], [3, 4]], [[5, 6], [6, 5]], [[4, 3], [5, 2]]]
            ).float(),
            "sum": torch.Tensor(
                [[[5, 0], [6, 4]], [[5, 8], [10, 8]], [[7, 4], [7, 3]]]
            ).float(),
            "prod": torch.Tensor(
                [[[4, -4], [9, 0]], [[0, 12], [24, 15]], [[12, 3], [10, 2]]]
            ).float(),
        }

        for method, expected_output in expected_output_for_method.items():
            model = make_fusion_layer(
                method, [self.fake_input_1.shape[-1], self.fake_input_2.shape[-1]]
            )
            output = model([self.fake_input_1, self.fake_input_2])
            self.assertTrue(torch.equal(output, expected_output))
            self.assertEqual(model.output_dim, self.fake_input_1.shape[-1])

    def test_concat_fusion(self):
        model = make_fusion_layer(
            "concat", [self.fake_input_1.shape[-1], self.fake_input_2.shape[-1]]
        )
        input_list = [self.fake_input_1, self.fake_input_2]
        output = model(input_list)
        expected_output = torch.cat(input_list, dim=-1)
        self.assertTrue(torch.equal(output, expected_output))

        expected_shape = self.fake_input_1.shape[-1] + self.fake_input_2.shape[-1]
        self.assertEqual(model.output_dim, expected_shape)

    def test_temporal_concat_fusion(self):
        model = make_fusion_layer(
            "temporal_concat",
            [self.fake_input_1.shape[-1], self.fake_input_2.shape[-1]],
        )
        input_list = [self.fake_input_1, self.fake_input_2]
        output = model(input_list)

        expected_output = torch.cat(input_list, dim=-2)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertEqual(model.output_dim, self.fake_input_2.shape[-1])

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest

import torch
import torch.nn as nn
from pytorchvideo.layers import make_multilayer_perceptron


class TestMLP(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_make_multilayer_perceptron(self):
        fake_input = torch.rand((8, 64))
        fcs = [64, 128, 64, 32]
        mid_activations = [nn.ReLU, nn.Sigmoid]
        final_activations = [nn.ReLU, nn.Sigmoid, None]
        norms = [nn.LayerNorm, nn.BatchNorm1d, None]
        for mid_act, final_act, norm in itertools.product(
            mid_activations, final_activations, norms
        ):
            mlp, output_dim = make_multilayer_perceptron(
                fully_connected_dims=fcs,
                mid_activation=mid_act,
                final_activation=final_act,
                norm=norm,
                dropout_rate=0.5,
            )

            self.assertEqual(output_dim, 32)

            output = mlp(fake_input)
            self.assertTrue(output.shape, torch.Size([8, 32]))

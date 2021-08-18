# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.layers import DropPath


class TestDropPath(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_dropPath(self):
        # Input should be same if drop_prob = 0.
        net_drop_path = DropPath(drop_prob=0.0)
        fake_input = torch.rand(64, 10, 20)
        output = net_drop_path(fake_input)
        self.assertTrue(output.equal(fake_input))
        # Test when drop_prob > 0.
        net_drop_path = DropPath(drop_prob=0.5)
        fake_input = torch.rand(64, 10, 20)
        output = net_drop_path(fake_input)
        self.assertTrue(output.shape, fake_input.shape)

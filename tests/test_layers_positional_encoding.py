# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import unittest

import torch
from pytorchvideo.layers import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

        self.batch_size = 4
        self.seq_len = 16
        self.feature_dim = 8
        self.fake_input = torch.randn(
            (self.batch_size, self.seq_len, self.feature_dim)
        ).float()
        lengths = torch.Tensor([16, 0, 14, 15, 16, 16, 16, 16])
        self.mask = torch.lt(
            torch.arange(self.seq_len)[None, :], lengths[:, None].long()
        )

    def test_positional_encoding(self):
        model = PositionalEncoding(self.feature_dim, self.seq_len)
        output = model(self.fake_input)
        delta = output - self.fake_input

        pe = torch.zeros(self.seq_len, self.feature_dim, dtype=torch.float)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2).float()
            * (-math.log(10000.0) / self.feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        for n in range(0, self.batch_size):
            self.assertTrue(torch.allclose(delta[n], pe, atol=1e-6))

    def test_positional_encoding_with_different_pe_and_data_dimensions(self):
        """Test that model executes even if input data dimensions
        differs from the dimension of initialized postional encoding model"""

        # When self.seq_len < positional_encoding_seq_len, pe is added to input
        positional_encoding_seq_len = self.seq_len * 3
        model = PositionalEncoding(self.feature_dim, positional_encoding_seq_len)
        output = model(self.fake_input)

        delta = output - self.fake_input
        pe = torch.zeros(self.seq_len, self.feature_dim, dtype=torch.float)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2).float()
            * (-math.log(10000.0) / self.feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        for n in range(0, self.batch_size):
            self.assertTrue(torch.allclose(delta[n], pe, atol=1e-6))

        # When self.seq_len > positional_encoding_seq_len, assertion error
        positional_encoding_seq_len = self.seq_len // 2
        model = PositionalEncoding(self.feature_dim, positional_encoding_seq_len)
        with self.assertRaises(AssertionError):
            output = model(self.fake_input)

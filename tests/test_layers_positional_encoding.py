# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import unittest

import torch
from pytorchvideo.layers import PositionalEncoding, SpatioTemporalClsPositionalEncoding


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

    def test_SpatioTemporalClsPositionalEncoding(self):
        # Test with cls token.
        batch_dim = 4
        dim = 16
        video_shape = (1, 2, 4)
        video_sum = math.prod(video_shape)
        has_cls = True
        model = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=video_shape,
            has_cls=has_cls,
        )
        fake_input = torch.rand(batch_dim, video_sum, dim)
        output = model(fake_input)
        output_gt_shape = (batch_dim, video_sum + 1, dim)
        self.assertEqual(tuple(output.shape), output_gt_shape)

        # Test without cls token.
        has_cls = False
        model = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=video_shape,
            has_cls=has_cls,
        )
        fake_input = torch.rand(batch_dim, video_sum, dim)
        output = model(fake_input)
        output_gt_shape = (batch_dim, video_sum, dim)
        self.assertEqual(tuple(output.shape), output_gt_shape)

        # Mismatch in dimension for patch_embed_shape.
        with self.assertRaises(AssertionError):
            model = SpatioTemporalClsPositionalEncoding(
                embed_dim=dim,
                patch_embed_shape=(1, 2),
                has_cls=has_cls,
            )

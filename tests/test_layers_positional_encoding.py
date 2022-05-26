# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
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
        video_sum = video_shape[0] * video_shape[1] * video_shape[2]
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

    def test_SpatioTemporalClsPositionalEncoding_nocls(self):
        # Test without cls token.
        batch_dim = 4
        dim = 16
        video_shape = (1, 2, 4)
        video_sum = video_shape[0] * video_shape[1] * video_shape[2]
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

    def test_SpatioTemporalClsPositionalEncoding_mismatch(self):
        # Mismatch in dimension for patch_embed_shape.
        with self.assertRaises(AssertionError):
            SpatioTemporalClsPositionalEncoding(
                embed_dim=16,
                patch_embed_shape=(1, 2),
            )

    def test_SpatioTemporalClsPositionalEncoding_scriptable(self):
        iter_embed_dim = [1, 2, 4, 32]
        iter_patch_embed_shape = [(1, 1, 1), (1, 2, 4), (32, 16, 1)]
        iter_sep_pos_embed = [True, False]
        iter_has_cls = [True, False]

        for (
            embed_dim,
            patch_embed_shape,
            sep_pos_embed,
            has_cls,
        ) in itertools.product(
            iter_embed_dim,
            iter_patch_embed_shape,
            iter_sep_pos_embed,
            iter_has_cls,
        ):
            stcpe = SpatioTemporalClsPositionalEncoding(
                embed_dim=embed_dim,
                patch_embed_shape=patch_embed_shape,
                sep_pos_embed=sep_pos_embed,
                has_cls=has_cls,
            )
            stcpe_scripted = torch.jit.script(stcpe)
            batch_dim = 4
            video_dim = math.prod(patch_embed_shape)
            fake_input = torch.rand(batch_dim, video_dim, embed_dim)
            expected = stcpe(fake_input)
            actual = stcpe_scripted(fake_input)
            torch.testing.assert_allclose(expected, actual)

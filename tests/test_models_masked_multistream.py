# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch
import torch.nn
from pytorchvideo.layers import PositionalEncoding, make_multilayer_perceptron
from pytorchvideo.models.masked_multistream import (
    LSTM,
    LearnMaskedDefault,
    MaskedSequential,
    MaskedTemporalPooling,
    TransposeMultiheadAttention,
    TransposeTransformerEncoder,
)


class TestMaskedMultiStream(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_masked_multistream_model(self):
        feature_dim = 8
        mlp, out_dim = make_multilayer_perceptron([feature_dim, 2])
        input_stream = MaskedSequential(
            PositionalEncoding(feature_dim),
            TransposeMultiheadAttention(feature_dim),
            MaskedTemporalPooling(method="avg"),
            torch.nn.LayerNorm(feature_dim),
            mlp,
            LearnMaskedDefault(out_dim),
        )

        seq_len = 10
        input_tensor = torch.rand([4, seq_len, feature_dim])
        mask = _lengths2mask(
            torch.tensor([seq_len, seq_len, seq_len, seq_len]), input_tensor.shape[1]
        )
        output = input_stream(input=input_tensor, mask=mask)
        self.assertEqual(output.shape, torch.Size([4, out_dim]))

    def test_masked_temporal_pooling(self):
        fake_input = torch.Tensor(
            [[[4, -2], [3, 0]], [[0, 2], [4, 3]], [[3, 1], [5, 2]]]
        ).float()
        valid_lengths = torch.Tensor([2, 1, 0]).int()
        valid_mask = _lengths2mask(valid_lengths, fake_input.shape[1])
        expected_output_for_method = {
            "max": torch.Tensor([[4, 0], [0, 2], [0, 0]]).float(),
            "avg": torch.Tensor([[3.5, -1], [0, 2], [0, 0]]).float(),
            "sum": torch.Tensor([[7, -2], [0, 2], [0, 0]]).float(),
        }
        for method, expected_output in expected_output_for_method.items():
            model = MaskedTemporalPooling(method)
            output = model(copy.deepcopy(fake_input), mask=valid_mask)
            self.assertTrue(torch.equal(output, expected_output))

    def test_transpose_attention(self):
        feature_dim = 8
        seq_len = 10
        fake_input = torch.rand([4, seq_len, feature_dim])
        mask = _lengths2mask(
            torch.tensor([seq_len, seq_len, seq_len, seq_len]), fake_input.shape[1]
        )
        model = TransposeMultiheadAttention(feature_dim, num_heads=2)
        output = model(fake_input, mask=mask)
        self.assertTrue(output.shape, fake_input.shape)

    def test_masked_lstm(self):
        feature_dim = 8
        seq_len = 10
        fake_input = torch.rand([4, seq_len, feature_dim])
        mask = _lengths2mask(
            torch.tensor([seq_len, seq_len, seq_len, seq_len]), fake_input.shape[1]
        )
        hidden_dim = 128

        model = LSTM(feature_dim, hidden_dim=hidden_dim, bidirectional=False)
        output = model(fake_input, mask=mask)
        self.assertTrue(output.shape, (fake_input.shape[0], hidden_dim))

        model = LSTM(feature_dim, hidden_dim=hidden_dim, bidirectional=True)
        output = model(fake_input, mask=mask)
        self.assertTrue(output.shape, (fake_input.shape[0], hidden_dim * 2))

    def test_masked_transpose_transformer_encoder(self):
        feature_dim = 8
        seq_len = 10
        fake_input = torch.rand([4, seq_len, feature_dim])
        mask = _lengths2mask(
            torch.tensor([seq_len, seq_len, seq_len, seq_len]), fake_input.shape[1]
        )

        model = TransposeTransformerEncoder(feature_dim)
        output = model(fake_input, mask=mask)
        self.assertEqual(output.shape, (fake_input.shape[0], feature_dim))

    def test_learn_masked_default(self):
        feature_dim = 8
        seq_len = 10
        fake_input = torch.rand([4, feature_dim])

        # All valid mask
        all_valid_mask = _lengths2mask(
            torch.tensor([seq_len, seq_len, seq_len, seq_len]), fake_input.shape[1]
        )
        model = LearnMaskedDefault(feature_dim)
        output = model(fake_input, mask=all_valid_mask)
        self.assertTrue(output.equal(fake_input))

        # No valid mask
        no_valid_mask = _lengths2mask(torch.tensor([0, 0, 0, 0]), fake_input.shape[1])
        model = LearnMaskedDefault(feature_dim)
        output = model(fake_input, mask=no_valid_mask)
        self.assertTrue(output.equal(model._learned_defaults.repeat(4, 1)))

        # Half valid mask
        half_valid_mask = _lengths2mask(torch.tensor([1, 1, 0, 0]), fake_input.shape[1])
        model = LearnMaskedDefault(feature_dim)
        output = model(fake_input, mask=half_valid_mask)
        self.assertTrue(output[:2].equal(fake_input[:2]))
        self.assertTrue(output[2:].equal(model._learned_defaults.repeat(2, 1)))


def _lengths2mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    return torch.lt(
        torch.arange(seq_len, device=lengths.device)[None, :], lengths[:, None].long()
    )

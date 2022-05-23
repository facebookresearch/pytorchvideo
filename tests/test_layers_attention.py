# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest

import torch
import torch.nn as nn
from pytorchvideo.layers import Mlp, MultiScaleAttention, MultiScaleBlock


class TestMLP(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_MultiScaleAttention(self):
        seq_len = 21
        c_dim = 10
        multiscale_attention = MultiScaleAttention(c_dim, num_heads=2)
        fake_input = torch.rand(8, seq_len, c_dim)
        input_shape = (2, 2, 5)
        output, output_shape = multiscale_attention(fake_input, input_shape)
        self.assertTrue(output.shape, fake_input.shape)

        # Test pooling kernel.
        multiscale_attention = MultiScaleAttention(
            c_dim,
            num_heads=2,
            stride_q=(2, 2, 1),
        )
        output, output_shape = multiscale_attention(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, 11, c_dim)
        gt_output_shape = (1, 1, 5)
        self.assertTrue(output.shape, gt_shape_tensor.shape)
        self.assertTrue(output_shape, gt_output_shape)

        # Test pooling kernel with no cls.
        seq_len = 20
        c_dim = 10
        fake_input = torch.rand(8, seq_len, c_dim)
        multiscale_attention = MultiScaleAttention(
            c_dim, num_heads=2, stride_q=(2, 2, 1), has_cls_embed=False
        )
        output, output_shape = multiscale_attention(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, int(seq_len / 2 / 2), c_dim)
        gt_output_shape = [1, 1, 5]
        self.assertEqual(output.shape, gt_shape_tensor.shape)
        self.assertEqual(output_shape, gt_output_shape)

    def test_MultiScaleBlock(self):
        # Change of output dimension.
        block = MultiScaleBlock(10, 20, 2)
        seq_len = 21
        c_dim = 10
        batch_dim = 8
        fake_input = torch.rand(batch_dim, seq_len, c_dim)
        input_shape = (2, 2, 5)
        output, output_shape = block(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, seq_len, 20)
        self.assertEqual(output.shape, gt_shape_tensor.shape)
        self.assertEqual(output_shape, input_shape)

        # Test pooling.
        block = MultiScaleBlock(10, 20, 2, stride_q=(2, 2, 1))
        c_dim = 10
        batch_dim = 8
        fake_input = torch.rand(batch_dim, seq_len, c_dim)
        input_shape = [2, 2, 5]
        output, output_shape = block(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, int((seq_len - 1) / 2 / 2) + 1, 20)
        gt_out_shape = [1, 1, 5]
        self.assertEqual(output.shape, gt_shape_tensor.shape)
        self.assertEqual(output_shape, gt_out_shape)

    def test_Mlp(self):
        fake_input = torch.rand((8, 64))
        in_features = [10, 20, 30]
        hidden_features = [10, 20, 20]
        out_features = [10, 20, 30]
        act_layers = [nn.GELU, nn.ReLU, nn.Sigmoid]
        drop_rates = [0.0, 0.1, 0.5]
        batch_size = 8
        for in_feat, hidden_feat, out_feat, act_layer, drop_rate in itertools.product(
            in_features, hidden_features, out_features, act_layers, drop_rates
        ):
            mlp_block = Mlp(
                in_features=in_feat,
                hidden_features=hidden_feat,
                out_features=out_feat,
                act_layer=act_layer,
                dropout_rate=drop_rate,
            )
            fake_input = torch.rand((batch_size, in_feat))
            output = mlp_block(fake_input)
            self.assertTrue(output.shape, torch.Size([batch_size, out_feat]))

    def test_MultiScaleBlock_is_scriptable(self):
        iter_qkv_bias = [True, False]
        iter_separate_qkv = [True, False]
        iter_dropout_rate = [0.0, 0.1]
        iter_droppath_rate = [0.0, 0.1]
        iter_norm_layer = [nn.LayerNorm]
        iter_attn_norm_layer = [nn.LayerNorm]
        iter_pool_mode = ["conv", "avg", "max"]
        iter_has_cls_embed = [True, False]
        iter_pool_first = [True, False]
        iter_residual_pool = [True, False]
        iter_depthwise_conv = [True, False]
        iter_bias_on = [True, False]
        iter_separate_qkv = [True, False]

        for (
            qkv_bias,
            dropout_rate,
            droppath_rate,
            norm_layer,
            attn_norm_layer,
            pool_mode,
            has_cls_embed,
            pool_first,
            residual_pool,
            depthwise_conv,
            bias_on,
            separate_qkv,
        ) in itertools.product(
            iter_qkv_bias,
            iter_dropout_rate,
            iter_droppath_rate,
            iter_norm_layer,
            iter_attn_norm_layer,
            iter_pool_mode,
            iter_has_cls_embed,
            iter_pool_first,
            iter_residual_pool,
            iter_depthwise_conv,
            iter_bias_on,
            iter_separate_qkv,
        ):
            msb = MultiScaleBlock(
                dim=10,
                dim_out=20,
                num_heads=2,
                stride_q=(2, 2, 1),
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
                droppath_rate=droppath_rate,
                norm_layer=norm_layer,
                attn_norm_layer=attn_norm_layer,
                pool_mode=pool_mode,
                has_cls_embed=has_cls_embed,
                pool_first=pool_first,
                residual_pool=residual_pool,
                depthwise_conv=depthwise_conv,
                bias_on=bias_on,
                separate_qkv=separate_qkv,
            )
            torch.jit.script(msb)

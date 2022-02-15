# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)


class TestFuseBN(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_fuse_bn(self):
        model = create_multiscale_vision_transformers(
            spatial_size=224,
            temporal_size=8,
            norm="batchnorm",
            embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            pool_kv_stride_adaptive=[1, 8, 8],
            pool_kvq_kernel=[3, 3, 3],
            cls_embed_on=False,
        )

        for blk in model.blocks:
            blk.norm1 = rand_init_bn(blk.norm1)
            blk.norm2 = rand_init_bn(blk.norm2)
            if blk.attn.norm_q:
                blk.attn.norm_q = rand_init_bn(blk.attn.norm_q)
            if blk.attn.norm_k:
                blk.attn.norm_k = rand_init_bn(blk.attn.norm_k)
            if blk.attn.norm_v:
                blk.attn.norm_v = rand_init_bn(blk.attn.norm_v)

        model.eval()

        x = torch.randn((4, 3, 8, 224, 224))
        expected_output = model(x)
        model.fuse_bn()
        output = model(x)
        self.assertTrue(torch.all(torch.isclose(output, expected_output, atol=1e-5)))
        self.assertTrue(
            len(
                [
                    layer
                    for layer in model.modules()
                    if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm3d))
                ]
            )
            == 0
        )


def rand_init_bn(bn):
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-0.5, 0.5)
    bn.running_var.data.uniform_(0.5, 1.5)
    bn.running_mean.data.uniform_(-0.5, 0.5)
    return bn

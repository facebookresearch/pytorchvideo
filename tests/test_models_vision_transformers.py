# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)


class TestVisionTransformers(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_create_mvit(self):
        """
        Test MViT.
        """
        # Test MViT with 3D case.
        num_head = 100
        batch_size = 4
        fake_input = torch.rand(batch_size, 3, 16, 224, 224)
        model = create_multiscale_vision_transformers(
            spatial_size=224,
            temporal_size=16,
            depth=2,
            head_num_classes=num_head,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)
        # Test MViT with 3D case with pool first.
        num_head = 100
        batch_size = 4
        fake_input = torch.rand(batch_size, 3, 16, 224, 224)
        model = create_multiscale_vision_transformers(
            spatial_size=224,
            temporal_size=16,
            depth=2,
            head_num_classes=num_head,
            pool_first=True,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)

        # Test MViT with 2D case for images.
        conv_patch_kernel = (7, 7)
        conv_patch_stride = (4, 4)
        conv_patch_padding = (3, 3)
        num_head = 100
        batch_size = 4
        fake_input = torch.rand(batch_size, 3, 224, 224)
        model = create_multiscale_vision_transformers(
            spatial_size=224,
            temporal_size=1,
            depth=2,
            head_num_classes=num_head,
            use_2d_patch=True,
            conv_patch_embed_kernel=conv_patch_kernel,
            conv_patch_embed_stride=conv_patch_stride,
            conv_patch_embed_padding=conv_patch_padding,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)

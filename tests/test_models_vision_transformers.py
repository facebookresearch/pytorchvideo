# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest
import warnings

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
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 4, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=28,
            temporal_size=4,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            pool_kv_stride_adaptive=[1, 2, 2],
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)
        # Test MViT with 3D case with pool first.
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 4, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=28,
            temporal_size=4,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            pool_first=True,
            pool_q_stride_size=[[0, 1, 2, 2]],
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)

        # Test MViT with 2D case for images.
        conv_patch_kernel = (7, 7)
        conv_patch_stride = (4, 4)
        conv_patch_padding = (3, 3)
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 3, 28, 28)
        model = create_multiscale_vision_transformers(
            spatial_size=(28, 28),
            temporal_size=1,
            patch_embed_dim=12,
            depth=1,
            head_num_classes=num_head,
            use_2d_patch=True,
            conv_patch_embed_kernel=conv_patch_kernel,
            conv_patch_embed_stride=conv_patch_stride,
            conv_patch_embed_padding=conv_patch_padding,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)

        # Test MViT without patch_embed.
        conv_patch_kernel = (7, 7)
        conv_patch_stride = (4, 4)
        conv_patch_padding = (3, 3)
        num_head = 100
        batch_size = 1
        fake_input = torch.rand(batch_size, 8, 12)
        model = create_multiscale_vision_transformers(
            spatial_size=(8, 1),
            temporal_size=1,
            patch_embed_dim=12,
            depth=1,
            enable_patch_embed=False,
            head_num_classes=num_head,
        )
        output = model(fake_input)
        gt_shape_tensor = torch.rand(batch_size, num_head)
        self.assertEqual(output.shape, gt_shape_tensor.shape)

        self.assertRaises(
            AssertionError,
            create_multiscale_vision_transformers,
            spatial_size=28,
            temporal_size=4,
            use_2d_patch=True,
        )

        self.assertRaises(
            AssertionError,
            create_multiscale_vision_transformers,
            spatial_size=28,
            temporal_size=1,
            pool_kv_stride_adaptive=[[2, 2, 2]],
            pool_kv_stride_size=[[1, 1, 2, 2]],
        )

        self.assertRaises(
            NotImplementedError,
            create_multiscale_vision_transformers,
            spatial_size=28,
            temporal_size=1,
            norm="fakenorm",
        )

    def test_mvit_is_torchscriptable(self):
        batch_size = 2
        num_head = 4
        spatial_size = (28, 28)
        temporal_size = 4
        depth = 2
        patch_embed_dim = 96

        # The following binary settings are covered by `test_layers_attention.py`:
        # `qkv_bias`, `depthwise_conv`, `separate_qkv`, `bias_on` `pool_first`
        # `residual_pool`
        true_false_opts = [
            "cls_embed_on",
            "sep_pos_embed",
            "enable_patch_embed",
            "enable_patch_embed_norm",
        ]

        # Loop over `2 ^ len(true_false_opts)` configurations
        for true_false_settings in itertools.product(
            *([[True, False]] * len(true_false_opts))
        ):
            named_tf_settings = dict(zip(true_false_opts, true_false_settings))

            model = create_multiscale_vision_transformers(
                spatial_size=spatial_size,
                temporal_size=temporal_size,
                depth=depth,
                head_num_classes=num_head,
                patch_embed_dim=patch_embed_dim,
                pool_kv_stride_adaptive=[1, 2, 2],
                **named_tf_settings,
                create_scriptable_model=False,
            ).eval()
            ts_model = torch.jit.script(model)

            input_shape = (
                (3, temporal_size, spatial_size[0], spatial_size[1])
                if named_tf_settings["enable_patch_embed"]
                else (
                    temporal_size * spatial_size[0] * spatial_size[1],
                    patch_embed_dim,
                )
            )
            fake_input = torch.rand(batch_size, *input_shape)

            expected = model(fake_input)
            actual = ts_model(fake_input)
            torch.testing.assert_allclose(expected, actual)

    def test_mvit_create_scriptable_model_is_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_multiscale_vision_transformers(
                spatial_size=28,
                temporal_size=4,
                norm="batchnorm",
                depth=2,
                head_num_classes=100,
                create_scriptable_model=True,
            )

        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)

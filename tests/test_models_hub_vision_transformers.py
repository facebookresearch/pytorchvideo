# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import unittest

import torch
import torch.nn as nn
from pytorchvideo.models.hub.utils import hub_model_builder


class TestHubVisionTransformers(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_load_hubconf(self):
        def test_load_mvit_(model_name, pretrained):
            path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
            )
            model = torch.hub.load(
                repo_or_dir=path,
                source="local",
                model=model_name,
                pretrained=pretrained,
            )
            self.assertIsNotNone(model)

        models = [
            "mvit_base_16x4",
            "mvit_base_16",
            "mvit_base_32x3",
        ]
        pretrains = [False, False, False]

        for model_name, pretrain in zip(models, pretrains):
            test_load_mvit_(model_name, pretrain)

    def test_hub_model_builder(self):
        def _fake_model(in_features=10, out_features=10) -> nn.Module:
            """
            A fake model builder with a linear layer.
            """
            model = nn.Linear(in_features, out_features)
            return model

        in_fea = 5
        default_config = {"in_features": in_fea}
        model = hub_model_builder(
            model_builder_func=_fake_model, default_config=default_config
        )
        self.assertEqual(model.in_features, in_fea)
        self.assertEqual(model.out_features, 10)

        # Test case where add_config overwrites default_config.
        in_fea = 5
        default_config = {"in_features": in_fea}
        add_in_fea = 2
        add_out_fea = 3

        model = hub_model_builder(
            model_builder_func=_fake_model,
            default_config=default_config,
            in_features=add_in_fea,
            out_features=add_out_fea,
        )
        self.assertEqual(model.in_features, add_in_fea)
        self.assertEqual(model.out_features, add_out_fea)

        # Test assertions.
        self.assertRaises(
            AssertionError,
            hub_model_builder,
            model_builder_func=_fake_model,
            pretrained=True,
            default_config={},
            fake_input=None,
        )

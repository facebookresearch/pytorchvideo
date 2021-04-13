# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.models.byol import BYOL
from torch import nn


class TestBYOL(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_byol(self):
        byol = BYOL(
            backbone=nn.Linear(8, 4),
            projector=nn.Linear(4, 4),
            feature_dim=4,
            norm=nn.BatchNorm1d,
        )
        for crop1, crop2 in TestBYOL._get_inputs():
            byol(crop1, crop2)

    @staticmethod
    def _get_inputs() -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = ((2, 8),)
        for shape in shapes:
            yield torch.rand(shape), torch.rand(shape)

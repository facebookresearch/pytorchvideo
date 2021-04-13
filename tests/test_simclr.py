# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.models.simclr import SimCLR
from torch import nn


class TestSimCLR(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_simclr(self):
        simclr = SimCLR(
            backbone=nn.Linear(8, 4),
            mlp=nn.Linear(4, 2),
            temperature=0.07,
        )
        for crop1, crop2 in TestSimCLR._get_inputs():
            simclr(crop1, crop2)

    @staticmethod
    def _get_inputs() -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = (
            (1, 8),
            (2, 8),
        )
        for shape in shapes:
            yield torch.rand(shape), torch.rand(shape)

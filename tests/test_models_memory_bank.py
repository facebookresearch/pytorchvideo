# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.models.memory_bank import MemoryBank
from torch import nn


class TestMemoryBank(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_memory_bank(self):
        simclr = MemoryBank(
            backbone=nn.Linear(8, 4),
            mlp=nn.Linear(4, 2),
            temperature=0.07,
            bank_size=8,
            dim=2,
        )
        for crop, ind in TestMemoryBank._get_inputs():
            simclr(crop, ind)

    @staticmethod
    def _get_inputs(bank_size: int = 8) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random inputs as test cases.
        shapes = ((2, 8),)
        for shape in shapes:
            yield torch.rand(shape), torch.randint(0, bank_size, size=(shape[0],))

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.mobile_cpu.convolutions import (
    Conv3d3x3x3DwBnRelu,
    Conv3dPwBnRelu,
)


class TestConv3dBlockEquivalency(unittest.TestCase):
    def test_Conv3dPwBnRelu_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 3, 4, 6, 6)
        # A conv block
        l0 = Conv3dPwBnRelu(3, 12)
        l1 = Conv3dPwBnRelu(
            12, 3, bias=True, use_relu=False
        )  # Skip relu to avoid NaN for rel error
        seq0 = nn.Sequential(l0, l1)
        seq0.eval()
        out0 = seq0(input_tensor)
        # Replicate the conv block
        l0_1 = deepcopy(l0)
        l1_1 = deepcopy(l1)
        # Convert into deployment mode
        l0_1.convert((1, 3, 4, 6, 6))  # Input tensor size is (1,3,4,6,6)
        l1_1.convert((1, 12, 4, 6, 6))  # Input tensor size is (1,12,4,6,6)
        seq1 = nn.Sequential(l0_1, l1_1)
        out1 = seq1(input_tensor)
        # Check arithmetic equivalency
        max_err = float(torch.max(torch.abs(out0 - out1)))
        rel_err = torch.abs((out0 - out1) / out0)
        max_rel_err = float(torch.max(rel_err))

        logging.info(
            (
                "test_Conv3dPwBnRelu_equivalency: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

    def test_Conv3d3x3x3DwBnRelu_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 3, 4, 6, 6)
        # A conv block
        l0 = Conv3dPwBnRelu(3, 12)
        l1 = Conv3d3x3x3DwBnRelu(12)
        l2 = Conv3dPwBnRelu(
            12, 3, bias=True, use_relu=False
        )  # Skip relu to avoid NaN for relative error
        seq0 = nn.Sequential(l0, l1, l2)
        seq0.eval()
        out0 = seq0(input_tensor)
        # Replicate the conv block
        l0_1 = deepcopy(l0)
        l1_1 = deepcopy(l1)
        l2_1 = deepcopy(l2)
        # Convert into deployment mode
        l0_1.convert((1, 3, 4, 6, 6))  # Input tensor size is (1,3,4,6,6)
        l1_1.convert((1, 12, 4, 6, 6))  # Input tensor size is (1,12,4,6,6)
        l2_1.convert((1, 12, 4, 6, 6))  # Input tensor size is (1,12,4,6,6)
        seq1 = nn.Sequential(l0_1, l1_1, l2_1)
        out1 = seq1(input_tensor)
        # Check arithmetic equivalency
        max_err = float(torch.max(torch.abs(out0 - out1)))
        rel_err = torch.abs((out0 - out1) / out0)
        max_rel_err = float(torch.max(rel_err))
        logging.info(
            (
                "test_Conv3d3x3x3DwBnRelu_equivalency: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

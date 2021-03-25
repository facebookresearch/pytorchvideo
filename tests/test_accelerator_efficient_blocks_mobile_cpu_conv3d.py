# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d3x1x1BnAct,
    Conv3d3x3x3DwBnAct,
    Conv3d5x1x1BnAct,
    Conv3dPwBnAct,
)


class TestConv3dBlockEquivalency(unittest.TestCase):
    def test_Conv3dPwBnAct_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 3, 4, 6, 6)
        # A conv block
        l0 = Conv3dPwBnAct(3, 12)
        l1 = Conv3dPwBnAct(
            12, 3, bias=True, activation="identity"
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
                "test_Conv3dPwBnAct_equivalency: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

    def test_Conv3d3x3x3DwBnAct_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 3, 4, 6, 6)
        # A conv block
        l0 = Conv3dPwBnAct(3, 12)
        l1 = Conv3d3x3x3DwBnAct(12)
        l2 = Conv3dPwBnAct(
            12, 3, bias=True, activation="identity"
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
                "test_Conv3d3x3x3DwBnAct_equivalency: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

    def test_Conv3d3x1x1BnAct_equivalency(self):
        for input_temporal in range(3):
            input_size = (1, 3, input_temporal + 1, 6, 6)
            # Input tensor
            input_tensor = torch.randn(input_size)
            # A conv block
            l0 = Conv3d3x1x1BnAct(3, 6)
            l0.eval()
            out0 = l0(input_tensor)
            # Replicate the conv block
            l0_1 = deepcopy(l0)
            # Convert into deployment mode
            l0_1.convert(input_size)  # Input tensor size is (1,3,4,6,6)
            out1 = l0_1(input_tensor)
            # Check output size
            assert (
                out0.size() == out1.size()
            ), f"Sizes of out0 {out0.size()} and out1 {out1.size()} are different."
            # Check arithmetic equivalency
            max_err = float(torch.max(torch.abs(out0 - out1)))
            rel_err = torch.abs((out0 - out1) / out0)
            max_rel_err = float(torch.max(rel_err))
            logging.info(
                (
                    "test_Conv3d3x1x1BnAct_equivalency: "
                    f"input tensor size: {input_size}"
                    f"max_err {max_err}, max_rel_err {max_rel_err}"
                )
            )
            self.assertTrue(max_err < 1e-3)

    def test_Conv3d5x1x1BnAct_equivalency(self):
        for input_temporal in range(5):
            input_size = (1, 3, input_temporal + 1, 6, 6)
            # Input tensor
            input_tensor = torch.randn(input_size)
            # A conv block
            l0 = Conv3d5x1x1BnAct(3, 6)
            l0.eval()
            out0 = l0(input_tensor)
            # Replicate the conv block
            l0_1 = deepcopy(l0)
            # Convert into deployment mode
            l0_1.convert(input_size)  # Input tensor size is (1,3,4,6,6)
            out1 = l0_1(input_tensor)
            # Check output size
            assert (
                out0.size() == out1.size()
            ), f"Sizes of out0 {out0.size()} and out1 {out1.size()} are different."
            # Check arithmetic equivalency
            max_err = float(torch.max(torch.abs(out0 - out1)))
            rel_err = torch.abs((out0 - out1) / out0)
            max_rel_err = float(torch.max(rel_err))
            logging.info(
                (
                    "test_Conv3d5x1x1BnAct_equivalency: "
                    f"input tensor size: {input_size}"
                    f"max_err {max_err}, max_rel_err {max_rel_err}"
                )
            )
            self.assertTrue(max_err < 1e-3)

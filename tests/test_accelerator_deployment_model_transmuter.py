# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

# Registers mobile_cpu transmuter functions
import pytorchvideo.accelerator.deployment.mobile_cpu.transmuter  # noqa: F401
import torch
import torch.nn as nn
from pytorchvideo.accelerator.deployment.common.model_transmuter import transmute_model
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)


class TestModelTransmuter(unittest.TestCase):
    def test_mobile_cpu_transmuter(self):
        # Input tensor
        input_blob_size = (1, 3, 2, 6, 6)
        input_tensor = torch.randn(input_blob_size)

        # Helper class to emulate user input model
        class _residual_block(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem0 = nn.Conv3d(3, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
                self.stem1 = nn.Conv3d(3, 3, kernel_size=(5, 1, 1), padding=(2, 0, 0))
                self.pw = nn.Conv3d(3, 6, kernel_size=1)
                self.relu = nn.ReLU()
                self.dw = nn.Conv3d(6, 6, kernel_size=3, padding=1, groups=6)
                self.relu1 = nn.ReLU()
                self.pwl = nn.Conv3d(6, 3, kernel_size=1)
                self.relu2 = nn.ReLU()

            def forward(self, x):
                out = self.stem0(x)
                out = self.stem1(out)
                out = self.pw(out)
                out = self.relu(out)
                out = self.dw(out)
                out = self.relu1(out)
                out = self.pwl(out)
                return self.relu2(out + x)

        user_model_ref = _residual_block()

        user_model_ref.eval()
        out_ref = user_model_ref(input_tensor)

        user_model_efficient = deepcopy(user_model_ref)
        transmute_model(
            user_model_efficient,
            target_device="mobile_cpu",
        )
        logging.info(f"after convert_model {user_model_efficient}")
        # Check whether blocks has been replaced by efficientBlock
        assert isinstance(user_model_efficient.pw, EfficientBlockBase), (
            f"user_model_efficient.pw {user_model_efficient.pw.__class__.__name__} "
            "is not converted!"
        )
        assert isinstance(user_model_efficient.dw, EfficientBlockBase), (
            f"user_model_efficient.dw {user_model_efficient.dw.__class__.__name__} "
            "is not converted!"
        )
        assert isinstance(user_model_efficient.pwl, EfficientBlockBase), (
            f"user_model_efficient.pwl {user_model_efficient.pwl.__class__.__name__} "
            "is not converted!"
        )
        user_model_efficient_converted = convert_to_deployable_form(
            user_model_efficient, input_tensor
        )
        out = user_model_efficient_converted(input_tensor)
        # Check arithmetic equivalency
        max_err = float(torch.max(torch.abs(out_ref - out)))
        rel_err = torch.abs((out_ref - out) / out_ref)
        max_rel_err = float(torch.max(rel_err))
        logging.info(
            (
                "test_mobile_cpu_transmuter: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

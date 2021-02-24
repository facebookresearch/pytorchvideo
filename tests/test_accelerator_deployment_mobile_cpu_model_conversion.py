# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from collections import OrderedDict

import torch
import torch.nn as nn
from pytorchvideo.accelerator.deployment.mobile_cpu.utils.model_conversion import (
    convert_to_deployable_form,
)
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)
from pytorchvideo.models.accelerator.mobile_cpu.residual_blocks import (
    X3dBottleneckBlock,
)


class TestDeploymentModelConversion(unittest.TestCase):
    def test_X3dBottleneckBlock_model_conversion(self):
        # Input tensor
        input_blob_size = (1, 3, 4, 6, 6)
        input_tensor = torch.randn(input_blob_size)

        # Helper class to emulate mix of efficient block and non efficient block
        class _quant_wrapper(nn.Module):
            # A common config where user model is wrapped by QuantStub/DequantStub
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()  # Non efficient block
                # X3dBottleneckBlock is efficient block consists of multiple efficient blocks
                self.model = X3dBottleneckBlock(
                    3,
                    12,
                    3,
                )
                self.dequant = torch.quantization.DeQuantStub()  # Non efficient block

            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x

        x3d_block_model_ref = _quant_wrapper()

        # Get ref output
        x3d_block_model_ref.eval()
        out_ref = x3d_block_model_ref(input_tensor)
        # Convert into deployment mode
        x3d_block_model_converted = convert_to_deployable_form(
            x3d_block_model_ref, input_tensor
        )
        out = x3d_block_model_converted(input_tensor)
        # Check arithmetic equivalency
        max_err = float(torch.max(torch.abs(out_ref - out)))
        rel_err = torch.abs((out_ref - out) / out_ref)
        max_rel_err = float(torch.max(rel_err))
        logging.info(
            (
                "test_X3dBottleneckBlock_model_conversion: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)
        # Check all sub-modules converted
        for iter_module in x3d_block_model_converted.modules():
            if isinstance(iter_module, EfficientBlockBase) and (
                hasattr(iter_module, "convert_flag")
            ):
                self.assertTrue(iter_module.convert_flag)
        # Check all hooks removed
        for iter_module in x3d_block_model_ref.modules():
            assert iter_module._forward_hooks == OrderedDict(), (
                f"{iter_module} in x3d_block_model_ref has non-empty _forward_hooks "
                f"{iter_module._forward_hooks}"
            )
        for iter_module in x3d_block_model_converted.modules():
            assert iter_module._forward_hooks == OrderedDict(), (
                f"{iter_module} in x3d_block_model_converted has non-empty _forward_hooks "
                f"{iter_module._forward_hooks}"
            )

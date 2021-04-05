# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

import torch
from pytorchvideo.models.accelerator.mobile_cpu.residual_blocks import (
    X3dBottleneckBlock,
)


class TestConv3dBlockEquivalency(unittest.TestCase):
    def test_X3dBottleneckBlock_equivalency(self):
        # Input tensor
        input_blob_size = (1, 3, 4, 6, 6)
        input_tensor = torch.randn(input_blob_size)
        for use_residual in (True, False):
            for spatial_stride in (1, 2):
                for se_ratio in (0, 0.5):
                    for act_func_0 in ("relu", "swish", "hswish", "identity"):
                        for act_func_1 in ("relu", "swish", "hswish", "identity"):
                            for act_func_2 in ("relu", "swish", "hswish", "identity"):
                                act_func_tuple = (act_func_0, act_func_1, act_func_2)
                                # X3dBottleneckBlock
                                x3d_block_ref = X3dBottleneckBlock(
                                    3,
                                    16,
                                    3,
                                    use_residual=use_residual,
                                    spatial_stride=spatial_stride,
                                    se_ratio=se_ratio,
                                    act_functions=act_func_tuple,
                                )
                                x3d_block = deepcopy(x3d_block_ref)
                                # Get ref output
                                x3d_block_ref.eval()
                                out_ref = x3d_block_ref(input_tensor)
                                # Convert into deployment mode
                                x3d_block.convert(input_blob_size)
                                out = x3d_block(input_tensor)
                                # Check arithmetic equivalency
                                max_err = float(torch.max(torch.abs(out_ref - out)))
                                rel_err = torch.abs((out_ref - out) / out_ref)
                                max_rel_err = float(torch.max(rel_err))
                                logging.info(
                                    (
                                        "test_X3dBottleneckBlock_equivalency: "
                                        f"current setting: use_residual {use_residual}, "
                                        f"spatial_stride {spatial_stride}, "
                                        f"se_ratio {se_ratio}, "
                                        f"act_func_tuple {act_func_tuple}, "
                                        f"max_err {max_err}, max_rel_err {max_rel_err}"
                                    )
                                )
                                self.assertTrue(max_err < 1e-3)

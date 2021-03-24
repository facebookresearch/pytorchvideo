# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

import torch
from pytorchvideo.layers.accelerator.mobile_cpu.fully_connected import FullyConnected
from pytorchvideo.layers.accelerator.mobile_cpu.pool import (
    AdaptiveAvgPool2d,
    AdaptiveAvgPool2dOutSize1,
    AdaptiveAvgPool3d,
    AdaptiveAvgPool3dOutSize1,
)


class TestHeadLayerEquivalency(unittest.TestCase):
    def test_head_layer_equivalency(self):
        for input_dim in (4, 5):  # 4 for BCHW, 5 for BCTHW
            input_tensor_size = (1, 3, 4, 6, 6) if input_dim == 5 else (1, 3, 6, 6)
            input_tensor = torch.randn(input_tensor_size)
            # Build up common head layer: pool + linear
            if input_dim == 5:
                pool_efficient_block_ref = AdaptiveAvgPool3d(1)
                pool_efficient_block_1 = AdaptiveAvgPool3d(1)
                pool_efficient_block_2 = AdaptiveAvgPool3dOutSize1()

            else:
                pool_efficient_block_ref = AdaptiveAvgPool2d(1)
                pool_efficient_block_1 = AdaptiveAvgPool2d(1)
                pool_efficient_block_2 = AdaptiveAvgPool2dOutSize1()
            pool_efficient_block_1.convert()
            pool_efficient_block_2.convert(input_tensor_size)
            linear_ref = FullyConnected(3, 8)
            linear_1 = deepcopy(linear_ref)
            linear_1.convert()

            ref_out = pool_efficient_block_ref(input_tensor)
            if input_dim == 5:
                ref_out = ref_out.permute((0, 2, 3, 4, 1))
            else:
                ref_out = ref_out.permute((0, 2, 3, 1))
            ref_out = linear_ref(ref_out)

            head_out_1 = pool_efficient_block_1(input_tensor)
            if input_dim == 5:
                head_out_1 = head_out_1.permute((0, 2, 3, 4, 1))
            else:
                head_out_1 = head_out_1.permute((0, 2, 3, 1))
            head_out_1 = linear_1(head_out_1)
            # Check arithmetic equivalency
            max_err = float(torch.max(torch.abs(ref_out - head_out_1)))
            rel_err = torch.abs((ref_out - head_out_1) / ref_out)
            max_rel_err = float(torch.max(rel_err))
            logging.info(
                (
                    "test_head_layer_equivalency: AdaptiveAvgPool + Linear"
                    f"input tensor size: {input_tensor_size}"
                    f"max_err {max_err}, max_rel_err {max_rel_err}"
                )
            )
            self.assertTrue(max_err < 1e-3)

            head_out_2 = pool_efficient_block_2(input_tensor)
            if input_dim == 5:
                head_out_2 = head_out_2.permute((0, 2, 3, 4, 1))
            else:
                head_out_2 = head_out_2.permute((0, 2, 3, 1))
            head_out_2 = linear_1(head_out_2)
            # Check arithmetic equivalency
            max_err = float(torch.max(torch.abs(ref_out - head_out_2)))
            rel_err = torch.abs((ref_out - head_out_2) / ref_out)
            max_rel_err = float(torch.max(rel_err))
            logging.info(
                (
                    "test_head_layer_equivalency: AdaptiveAvgPoolOutSize1 + Linear"
                    f"input tensor size: {input_tensor_size}"
                    f"max_err {max_err}, max_rel_err {max_rel_err}"
                )
            )
            self.assertTrue(max_err < 1e-3)

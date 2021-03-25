# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from copy import deepcopy

import torch
from pytorchvideo.layers.accelerator.mobile_cpu.activation_functions import (
    supported_act_functions,
)
from pytorchvideo.layers.accelerator.mobile_cpu.attention import SqueezeExcitation


class TestActivationAttentionEquivalency(unittest.TestCase):
    def test_activation_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 3, 4, 6, 6)
        for iter_activation_name in supported_act_functions:
            act_func_ref = supported_act_functions[iter_activation_name]()
            act_func_convert = deepcopy(act_func_ref)
            act_func_convert.convert()
            # Get output of both activations
            out0 = act_func_ref(input_tensor)
            out1 = act_func_convert(input_tensor)
            # Check arithmetic equivalency
            max_err = float(torch.max(torch.abs(out0 - out1)))

            logging.info(
                f"test_activation_equivalency: {iter_activation_name} max_err {max_err}"
            )
            self.assertTrue(max_err < 1e-3)

    def test_squeeze_excite_equivalency(self):
        # Input tensor
        input_tensor = torch.randn(1, 16, 4, 6, 6)
        # Instantiate ref and convert se modules.
        se_ref = SqueezeExcitation(16, num_channels_reduced=2, is_3d=True)
        se_ref.eval()
        se_convert = deepcopy(se_ref)
        se_convert.convert((1, 16, 4, 6, 6))
        # Get output of both activations
        out0 = se_ref(input_tensor)
        out1 = se_convert(input_tensor)
        # Check arithmetic equivalency
        max_err = float(torch.max(torch.abs(out0 - out1)))
        rel_err = torch.abs((out0 - out1) / out0)
        max_rel_err = float(torch.max(rel_err))

        logging.info(
            (
                "test_squeeze_excite_equivalency: "
                f"max_err {max_err}, max_rel_err {max_rel_err}"
            )
        )
        self.assertTrue(max_err < 1e-3)

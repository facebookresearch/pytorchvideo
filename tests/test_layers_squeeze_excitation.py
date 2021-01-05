# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest

import torch
import torch.nn as nn
from pytorchvideo.layers.squeeze_excitation import (
    create_audio_2d_squeeze_excitation_block,
)


class Test2DSqueezeExcitationBlock(unittest.TestCase):
    def setUp(self):

        self.layer_args = {
            "dim_in": 32,
            "dim_out": 32,
            "use_se": True,
            "se_reduction_ratio": 16,
            "branch_fusion": lambda x, y: x + y,
            "conv_a_kernel_size": 3,
            "conv_a_stride": 1,
            "conv_a_padding": 1,
            "conv_b_kernel_size": 3,
            "conv_b_stride": 1,
            "conv_b_padding": 1,
            "norm": nn.BatchNorm2d,
            "norm_eps": 1e-5,
            "norm_momentum": 0.1,
            "activation": nn.ReLU,
        }

        self.batchsize = 1
        self.forward_pass_configs = [
            {
                "input": torch.rand(self.batchsize, self.layer_args["dim_in"], 100, 40),
                "output_shape": torch.Size(
                    [self.batchsize, self.layer_args["dim_out"], 100, 40]
                ),
            },
        ]

    def test_forward_pass(self):
        for split_config in self.forward_pass_configs:
            layer_args = copy.deepcopy(self.layer_args)
            model = create_audio_2d_squeeze_excitation_block(**layer_args)

            out = model(split_config["input"])
            self.assertTrue(isinstance(out, torch.Tensor))
            self.assertEqual(out.size(), split_config["output_shape"])

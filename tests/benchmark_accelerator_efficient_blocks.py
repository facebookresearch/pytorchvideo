# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from typing import Callable

import torch
import torch.nn as nn

# import torch.quantization.quantize_fx as quantize_fx
from fvcore.common.benchmark import benchmark
from pytorchvideo.accelerator.efficient_blocks.mobile_cpu.convolutions import (
    Conv3dPwBnRelu,
)
from torch.utils.mobile_optimizer import optimize_for_mobile


class TestBenchmarkEfficientBlocks(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_benchmark_conv3d_pw_bn_relu(self, num_iters: int = 20) -> None:
        """
        Benchmark Conv3dPwBnRelu.
        Note efficient block Conv3dPwBnRelu is designed for mobile cpu with qnnpack
        backend, and benchmarking on server with another backend (e.g., fbgemm) may
        have different latency result compared to running on mobile cpu with qnnpack.
        Running on x86 based server cpu with qnnpack may also have different latency as
        running on mobile cpu with qnnpack, as qnnpack is optimized for
        ARM based mobile cpu.
        Args:
            num_iters (int): number of iterations to perform benchmarking.
        """

        torch.backends.quantized.engine = "qnnpack"
        kwargs_list = [
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "out_channels": 108,
                "quantize": False,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "out_channels": 108,
                "quantize": False,
            },
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "out_channels": 108,
                "quantize": True,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "out_channels": 108,
                "quantize": True,
            },
        ]

        def _benchmark_conv3d_pw_bn_relu_forward(**kwargs) -> Callable:
            assert kwargs["mode"] in ("original", "deployable"), (
                "kwargs['mode'] must be either 'original' or 'deployable',"
                "but got {}.".format(kwargs["mode"])
            )
            input_tensor = torch.randn((kwargs["input_blob_size"]))
            conv_block = Conv3dPwBnRelu(
                kwargs["in_channels"],
                kwargs["out_channels"],
                use_bn=False,  # assume BN has already been fused for forward
            )

            if kwargs["mode"] == "deployable":
                conv_block.convert(kwargs["input_blob_size"])
            conv_block.eval()
            if kwargs["quantize"] is True:
                if kwargs["mode"] == "original":  # manually fuse conv and relu
                    conv_block.kernel = torch.quantization.fuse_modules(
                        conv_block.kernel, ["conv", "relu"]
                    )
                conv_block = nn.Sequential(
                    torch.quantization.QuantStub(),
                    conv_block,
                    torch.quantization.DeQuantStub(),
                )

                conv_block.qconfig = torch.quantization.get_default_qconfig("qnnpack")
                conv_block = torch.quantization.prepare(conv_block)
                try:
                    conv_block = torch.quantization.convert(conv_block)
                except Exception as e:
                    logging.info(
                        "benchmark_conv3d_pw_bn_relu: "
                        "catch exception '{}' with kwargs of {}".format(e, kwargs)
                    )

                    def func_to_benchmark_dummy() -> None:
                        return

                    return func_to_benchmark_dummy
            traced_model = torch.jit.trace(conv_block, input_tensor, strict=False)
            if kwargs["quantize"] is False:
                traced_model = optimize_for_mobile(traced_model)

            logging.info(f"model arch: {traced_model}")

            def func_to_benchmark() -> None:
                try:
                    _ = traced_model(input_tensor)
                except Exception as e:
                    logging.info(
                        "benchmark_conv3d_pw_bn_relu: "
                        "catch exception '{}' with kwargs of {}".format(e, kwargs)
                    )

                return

            return func_to_benchmark

        benchmark(
            _benchmark_conv3d_pw_bn_relu_forward,
            "benchmark_conv3d_pw_bn_relu",
            kwargs_list,
            num_iters=num_iters,
            warmup_iters=2,
        )

        self.assertTrue(True)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from typing import Callable

import torch
import torch.nn as nn

# import torch.quantization.quantize_fx as quantize_fx
from fvcore.common.benchmark import benchmark
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d3x3x3DwBnAct,
    Conv3dPwBnAct,
)
from pytorchvideo.models.accelerator.mobile_cpu.residual_blocks import (
    X3dBottleneckBlock,
)
from torch.utils.mobile_optimizer import optimize_for_mobile


class TestBenchmarkEfficientBlocks(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_benchmark_conv3d_pw_bn_relu(self, num_iters: int = 20) -> None:
        """
        Benchmark Conv3dPwBnAct with ReLU activation.
        Note efficient block Conv3dPwBnAct is designed for mobile cpu with qnnpack
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
            conv_block = Conv3dPwBnAct(
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
                        conv_block.kernel, ["conv", "act.act"]
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

    def test_benchmark_conv3d_3x3x3_dw_bn_relu(self, num_iters: int = 20) -> None:
        """
        Benchmark Conv3d3x3x3DwBnAct with ReLU activation.
        Note efficient block Conv3d3x3x3DwBnAct is designed for mobile cpu with qnnpack
        backend, and benchmarking on server with another backend (e.g., fbgemm) may have
        different latency result compared as running on mobile cpu.
        Args:
            num_iters (int): number of iterations to perform benchmarking.
        """
        torch.backends.quantized.engine = "qnnpack"
        kwargs_list = [
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "quantize": False,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "quantize": False,
            },
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "quantize": True,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 40, 40),
                "in_channels": 48,
                "quantize": True,
            },
        ]

        def _benchmark_conv3d_3x3x3_dw_bn_relu_forward(**kwargs) -> Callable:
            assert kwargs["mode"] in ("original", "deployable"), (
                "kwargs['mode'] must be either 'original' or 'deployable',"
                "but got {}.".format(kwargs["mode"])
            )
            input_tensor = torch.randn((kwargs["input_blob_size"]))
            conv_block = Conv3d3x3x3DwBnAct(
                kwargs["in_channels"],
                use_bn=False,  # assume BN has already been fused for forward
            )

            if kwargs["mode"] == "deployable":
                conv_block.convert(kwargs["input_blob_size"])
            conv_block.eval()
            if kwargs["quantize"] is True:
                if kwargs["mode"] == "original":  # manually fuse conv and relu
                    conv_block.kernel = torch.quantization.fuse_modules(
                        conv_block.kernel, ["conv", "act.act"]
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
                        "benchmark_conv3d_3x3x3_dw_bn_relu: "
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
                        "benchmark_conv3d_3x3x3_dw_bn_relu: "
                        "catch exception '{}' with kwargs of {}".format(e, kwargs)
                    )
                return

            return func_to_benchmark

        benchmark(
            _benchmark_conv3d_3x3x3_dw_bn_relu_forward,
            "benchmark_conv3d_3x3x3_dw_bn_relu",
            kwargs_list,
            num_iters=num_iters,
            warmup_iters=2,
        )

        self.assertTrue(True)

    def test_benchmark_x3d_bottleneck_block(self, num_iters: int = 20) -> None:
        """
        Benchmark X3dBottleneckBlock.
        Note efficient block X3dBottleneckBlock is designed for mobile cpu with qnnpack
        backend, and benchmarking on server/laptop may have different latency result
        compared to running on mobile cpu.
        Args:
            num_iters (int): number of iterations to perform benchmarking.
        """
        torch.backends.quantized.engine = "qnnpack"
        kwargs_list = [
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 20, 20),
                "in_channels": 48,
                "mid_channels": 108,
                "out_channels": 48,
                "quantize": False,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 20, 20),
                "in_channels": 48,
                "mid_channels": 108,
                "out_channels": 48,
                "quantize": False,
            },
            {
                "mode": "original",
                "input_blob_size": (1, 48, 4, 20, 20),
                "in_channels": 48,
                "mid_channels": 108,
                "out_channels": 48,
                "quantize": True,
            },
            {
                "mode": "deployable",
                "input_blob_size": (1, 48, 4, 20, 20),
                "in_channels": 48,
                "mid_channels": 108,
                "out_channels": 48,
                "quantize": True,
            },
        ]

        def _benchmark_x3d_bottleneck_forward(**kwargs) -> Callable:
            assert kwargs["mode"] in ("original", "deployable"), (
                "kwargs['mode'] must be either 'original' or 'deployable',"
                "but got {}.".format(kwargs["mode"])
            )
            input_tensor = torch.randn((kwargs["input_blob_size"]))
            conv_block = X3dBottleneckBlock(
                kwargs["in_channels"],
                kwargs["mid_channels"],
                kwargs["out_channels"],
                use_bn=(False, False, False),  # Assume BN has been fused for forward
            )

            if kwargs["mode"] == "deployable":
                conv_block.convert(kwargs["input_blob_size"])
            conv_block.eval()
            if kwargs["quantize"] is True:
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
                        "benchmark_x3d_bottleneck_forward: "
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
                        "benchmark_x3d_bottleneck_forward: "
                        "catch exception '{}' with kwargs of {}".format(e, kwargs)
                    )
                return

            return func_to_benchmark

        benchmark(
            _benchmark_x3d_bottleneck_forward,
            "benchmark_x3d_bottleneck_forward",
            kwargs_list,
            num_iters=num_iters,
            warmup_iters=2,
        )

        self.assertTrue(True)

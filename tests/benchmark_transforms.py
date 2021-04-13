# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
from typing import Callable

import torch
from fvcore.common.benchmark import benchmark
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms.functional import short_side_scale
from utils import create_dummy_video_frames


class TestBenchmarkTransforms(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_benchmark_short_side_scale_pytorch(self, num_iters: int = 10) -> None:
        """
        Benchmark scale operation with pytorch backend.
        Args:
            num_iters (int): number of iterations to perform benchmarking.
        """
        kwargs_list = [
            {"temporal_size": 8, "ori_spatial_size": (128, 128), "dst_short_size": 112},
            {
                "temporal_size": 16,
                "ori_spatial_size": (128, 128),
                "dst_short_size": 112,
            },
            {
                "temporal_size": 32,
                "ori_spatial_size": (128, 128),
                "dst_short_size": 112,
            },
            {"temporal_size": 8, "ori_spatial_size": (256, 256), "dst_short_size": 224},
            {
                "temporal_size": 16,
                "ori_spatial_size": (256, 256),
                "dst_short_size": 224,
            },
            {
                "temporal_size": 32,
                "ori_spatial_size": (256, 256),
                "dst_short_size": 224,
            },
            {"temporal_size": 8, "ori_spatial_size": (320, 320), "dst_short_size": 224},
            {
                "temporal_size": 16,
                "ori_spatial_size": (320, 320),
                "dst_short_size": 224,
            },
            {
                "temporal_size": 32,
                "ori_spatial_size": (320, 320),
                "dst_short_size": 224,
            },
        ]

        def _init_benchmark_short_side_scale(**kwargs) -> Callable:
            x = thwc_to_cthw(
                create_dummy_video_frames(
                    kwargs["temporal_size"],
                    kwargs["ori_spatial_size"][0],
                    kwargs["ori_spatial_size"][1],
                )
            ).to(dtype=torch.float32)

            def func_to_benchmark() -> None:
                _ = short_side_scale(x, kwargs["dst_short_size"])
                return

            return func_to_benchmark

        benchmark(
            _init_benchmark_short_side_scale,
            "benchmark_short_side_scale_pytorch",
            kwargs_list,
            num_iters=num_iters,
            warmup_iters=2,
        )
        self.assertTrue(True)

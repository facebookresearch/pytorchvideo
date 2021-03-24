# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
from pytorchvideo.layers.distributed import DifferentiableAllGather
from torch.multiprocessing import Process


class TestDistributedOps(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    @unittest.skipIf(
        not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
        "The current machine does not has more than two devices to perform all gather",
    )
    def test_all_gather_with_gradient(self):
        tensor_list = [
            torch.tensor([[1], [1]]).to("cuda:0"),
            torch.tensor([[2], [2]]).to("cuda:1"),
        ]
        expected_output = torch.tensor([[1], [1], [2], [2]])

        def run_distributed(rank, size):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            dist.init_process_group("gloo", rank=rank, world_size=size)
            output = DifferentiableAllGather.apply(tensor_list[rank])
            self.assertTrue(
                np.allclose(
                    output.numpy(), expected_output.numpy(), rtol=1e-1, atol=1e-1
                )
            )

        num_devices = 2
        processes = []
        for rank in range(num_devices):
            p = Process(target=run_distributed, args=(rank, num_devices))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

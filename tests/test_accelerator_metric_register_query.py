# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import unittest
from functools import reduce

import torch
import torch.nn as nn
from pytorchvideo.accelerator.metric_engine.metric_engine import (
    METRIC_QUERY_REGISTRY,
    metric_query,
)


@METRIC_QUERY_REGISTRY.register()
def simple_conv3d_flops(model: nn.Conv3d, input_data: torch.Tensor, **kwargs) -> int:
    """
    Registers a simple conv3d flops as metric query function.
    Args:
        model (nn.Conv3d): the conv3d layer for query
        input_data (torch.Tensor): the input data to go through conv3d layer for query
        kwargs (dict): any other keyword arguments
    Return:
        flop (int): flops of the model with input_data
    """
    output_blob_size = reduce(lambda x, y: x * y, model(input_data).size())
    flops = (
        output_blob_size
        * reduce(lambda x, y: x * y, model.kernel_size)
        * model.in_channels
        / reduce(lambda x, y: x * y, model.stride)
        / model.groups
    )
    return int(flops)


class TestMetricRegisterQuery(unittest.TestCase):
    def test_simple3d_flops(self):
        test_model = nn.Conv3d(16, 32, kernel_size=1, stride=1, groups=1)
        input_data = torch.randn(1, 16, 4, 16, 16)
        flops = metric_query(
            test_model, input_data, metric_query_name="simple_conv3d_flops"
        )
        logging.info(f"TestMetricRegisterQuery - flops: {flops}")

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from pytorchvideo.layers.swish import Swish
from pytorchvideo.layers.utils import round_width


class SE(nn.Module):
    """
    Implementation of the Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC,
    Swish, FC, Sigmoid.

    Squeeze-and-excitation networks. Hu, Jie and Shen, Li and Sun, Gang. CVPR 2018
    """

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
            divisor (int): the new width should be dividable by divisor.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x

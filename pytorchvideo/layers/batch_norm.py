# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.distributed as dist
from fvcore.nn.distributed import differentiable_all_reduce
from pytorchvideo.layers.distributed import get_world_size
from torch import nn


class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """
    An implementation of 1D naive sync batch normalization. See details in
    NaiveSyncBatchNorm2d below.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        vec = torch.cat([mean, meansqr], dim=0)
        vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)

        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        return input * scale + bias


class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """
    An implementation of 2D naive sync batch normalization.
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        This module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        vec = torch.cat([mean, meansqr], dim=0)
        vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        return input * scale + bias


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):

    """
    An implementation of 3D naive sync batch normalization. See details in
    NaiveSyncBatchNorm2d above.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        vec = torch.cat([mean, meansqr], dim=0)
        vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)

        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        return input * scale + bias

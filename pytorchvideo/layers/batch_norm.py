# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pytorchvideo.layers.distributed as du
import torch
import torch.distributed as dist
from fvcore.nn.distributed import differentiable_all_reduce
from torch import nn


class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """
    An implementation of 1D naive sync batch normalization. See details in
    NaiveSyncBatchNorm2d below.

    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):

        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(
                f"Cannot set num_sync_devices separately when global_sync = {self.global_sync}"
            )
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(
                f"num_sync_devices cannot be None when global_sync = {self.global_sync}"
            )

        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (
                    du.get_local_size(),
                    self.num_sync_devices,
                )
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm1d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])

        vec = torch.cat([mean, meansqr], dim=0)
        #  sync stats globally or locally
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
                1.0 / self.num_sync_devices
            )

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)

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

    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.

    Note:
        This module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):

        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(
                f"Cannot set num_sync_devices separately when global_sync = {self.global_sync}"
            )
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(
                f"num_sync_devices cannot be None when global_sync = {self.global_sync}"
            )

        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (
                    du.get_local_size(),
                    self.num_sync_devices,
                )
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm2d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        #  sync stats globally or locally
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
                1.0 / self.num_sync_devices
            )

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
    Naive version of Synchronized 3D BatchNorm. See details in
    NaiveSyncBatchNorm2d above.
    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):

        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(
                f"Cannot set num_sync_devices separately when global_sync = {self.global_sync}"
            )
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(
                f"num_sync_devices cannot be None when global_sync = {self.global_sync}"
            )

        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (
                    du.get_local_size(),
                    self.num_sync_devices,
                )
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        assert B > 0, "SyncBatchNorm does not support zero batch size."

        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        #  sync stats globally or locally
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
                1.0 / self.num_sync_devices
            )

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

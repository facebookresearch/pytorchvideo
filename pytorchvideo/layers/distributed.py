# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup
from torch.autograd.function import Function

_LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    """
    Simple wrapper for correctly getting worldsize in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )


def cat_all_gather(tensors, local=False):
    """Performs the concatenated all_reduce operation on the provided tensors."""
    if local:
        gather_sz = get_local_size()
    else:
        gather_sz = torch.distributed.get_world_size()
    tensors_gather = [torch.ones_like(tensors) for _ in range(gather_sz)]
    torch.distributed.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
        group=_LOCAL_PROCESS_GROUP if local else None,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def init_distributed_training(cfg):
    """
    Initialize variables needed for distributed training.
    """
    if cfg.NUM_GPUS <= 1:
        return
    num_gpus_per_machine = cfg.NUM_GPUS
    num_machines = dist.get_world_size() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == cfg.SHARD_ID:
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_process_group() -> ProcessGroup:
    assert _LOCAL_PROCESS_GROUP is not None
    return _LOCAL_PROCESS_GROUP


class GroupGather(Function):
    """
    GroupGather performs all gather on each of the local process/ GPU groups.
    """

    @staticmethod
    def forward(ctx, input, num_sync_devices, num_groups):
        """
        Perform forwarding, gathering the stats across different process/ GPU
        group.
        """
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups

        input_list = [torch.zeros_like(input) for k in range(get_local_size())]
        dist.all_gather(
            input_list, input, async_op=False, group=get_local_process_group()
        )

        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[
                group_idx * num_sync_devices : (group_idx + 1) * num_sync_devices
            ]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform backwarding, gathering the gradients across different process/ GPU
        group.
        """
        grad_output_list = [
            torch.zeros_like(grad_output) for k in range(get_local_size())
        ]
        dist.all_gather(
            grad_output_list,
            grad_output,
            async_op=False,
            group=get_local_process_group(),
        )

        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[
                group_idx
                * ctx.num_sync_devices : (group_idx + 1)
                * ctx.num_sync_devices
            ]
        grads = torch.sum(grads, dim=0)
        return grads, None, None

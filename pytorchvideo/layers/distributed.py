# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.distributed as dist


class DifferentiableAllGather(torch.autograd.Function):
    """
    The torch.distributed.all_gather function does not back propagate the gradient back
    to different devices. DifferentiableAllGather is a operator that perform all_gather
    and back propagate the gradient and bring them back to all devices.
    """

    @staticmethod
    def forward(ctx, input):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            x_gather = [torch.ones_like(input) for _ in range(world_size)]
            dist.all_gather(x_gather, input, async_op=False)
            input = torch.cat(x_gather, dim=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_available() and dist.is_initialized():
            reduction = dist.all_reduce(grad_output, async_op=True)
            reduction.wait()
            world_size = dist.get_world_size()
            N = grad_output.size(0)
            mini_batchsize = N // world_size
            cur_gpu = dist.get_rank()
            grad_output = grad_output[
                cur_gpu * mini_batchsize : (cur_gpu + 1) * mini_batchsize
            ]
        return grad_output


class DifferentiableAllReduce(torch.autograd.Function):
    """
    The torch.distributed.all_reduce gathers and reduces tensors in-place in one step.
    This implementation uses torch.distributed.all_gather to first gather separate tensors
    and explicitly reduce them afterwards. It is more reliable but maybe less efficient.
    """

    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


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

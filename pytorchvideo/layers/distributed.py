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

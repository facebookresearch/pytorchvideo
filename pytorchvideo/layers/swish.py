# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    """

    def forward(self, x):
        return SwishFunction.apply(x)


class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).

    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

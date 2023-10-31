# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x).

    Swish is a non-linear activation function that has shown promising results in
    neural network architectures. It is defined as the element-wise product of
    the input tensor and the sigmoid of the input tensor.

    References:
    - "Searching for activation functions" by Prajit Ramachandran, Barret Zoph, and Quoc V. Le (2017)

    Example:
    ```python
    activation = Swish()
    output = activation(input_tensor)
    ```

    Note:
    - The Swish function has been found to be effective in various deep learning tasks.
    - It is differentiable and often produces smoother gradients compared to ReLU.

    Args:
    None

    Returns:
    torch.Tensor: The tensor after applying the Swish activation.

    Shape:
    - Input: Any shape as long as it is broadcastable to the output shape.
    - Output: Same shape as the input.

    Examples:
    >>> activation = Swish()
    >>> input_tensor = torch.tensor([1.0, 2.0, 3.0])
    >>> output = activation(input_tensor)
    >>> output
    tensor([0.7311, 1.7616, 2.9466])
    ```

    """

    def forward(self, x):
        return SwishFunction.apply(x)


class SwishFunction(torch.autograd.Function):
    """
    Autograd function for the Swish activation.

    Args:
    - ctx (context): A context object to save information for backward pass.
    - x (Tensor): The input tensor.

    Returns:
    - result (Tensor): The output tensor after applying the Swish activation.
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
    
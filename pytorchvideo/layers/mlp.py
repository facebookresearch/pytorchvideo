# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Callable, List, Optional, Tuple

from torch import nn


def make_multilayer_perceptron(
    fully_connected_dims: List[int],
    norm: Optional[Callable] = None,
    mid_activation: Callable = nn.ReLU,
    final_activation: Optional[Callable] = nn.ReLU,
    dropout_rate: float = 0.0,
) -> Tuple[nn.Module, int]:
    """
    Create a Multi-Layer Perceptron (MLP) with customizable architecture.

    Args:
        fully_connected_dims (List[int]): A list of integers specifying the dimensions of
            fully connected layers in the MLP. The list should have at least two elements,
            where the first element is the input dimension and the last element is the output
            dimension.

        norm (Optional[Callable]): A callable normalization function to be applied after each
            fully connected layer (e.g., nn.BatchNorm1d). If None, no normalization is applied.

        mid_activation (Callable): A callable activation function to be applied after each
            fully connected layer except the last one (e.g., nn.ReLU).

        final_activation (Optional[Callable]): A callable activation function to be applied
            after the last fully connected layer. If None, no activation is applied after
            the final layer.

        dropout_rate (float): The dropout rate to be applied after each fully connected layer.
            If 0.0, no dropout is applied.

    Returns:
        Tuple[nn.Module, int]: A tuple containing the MLP module and the output dimension.

    Example:
        To create a simple MLP with two hidden layers of size 64 and 32:
        ```
        mlp, output_dim = make_multilayer_perceptron(
            fully_connected_dims=[input_dim, 64, 32, output_dim],
            norm=nn.BatchNorm1d,
            mid_activation=nn.ReLU,
            final_activation=nn.Sigmoid,
            dropout_rate=0.1
        )
        ```

    Note:
        - The `fully_connected_dims` list must have at least two elements, with the first
          element representing the input dimension and the last element representing the output
          dimension.
        - You can customize the architecture of the MLP by specifying the number of hidden layers
          and their dimensions.
        - Activation functions are applied after each hidden layer, except for the final layer.
        - If `norm` is provided, it is applied after each hidden layer.
        - If `dropout_rate` is greater than 0.0, dropout is applied after each hidden layer.
        - The final activation function is applied after the last hidden layer if provided.
    """
    assert isinstance(fully_connected_dims, list)
    assert len(fully_connected_dims) > 1
    assert all(_is_pos_int(x) for x in fully_connected_dims)

    layers = []
    cur_dim = fully_connected_dims[0]
    for dim in fully_connected_dims[1:-1]:
        layers.append(nn.Linear(cur_dim, dim))
        if norm is not None:
            layers.append(norm(dim))
        layers.append(mid_activation())
        cur_dim = dim
    layers.append(nn.Linear(cur_dim, fully_connected_dims[-1]))
    if dropout_rate > 0:
        layers.append(nn.Dropout(p=dropout_rate))
    if final_activation is not None:
        layers.append(final_activation())

    mlp = nn.Sequential(*layers)
    output_dim = fully_connected_dims[-1]
    return mlp, output_dim


def _is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0

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
    Factory function for Multi-Layer Perceptron. These are constructed as repeated
    blocks of the following format where each fc represents the blocks output/input dimension.

    ::

                             Linear (in=fc[i-1], out=fc[i])
                                           ↓
                                 Normalization (norm)
                                           ↓
                               Activation (mid_activation)
                                           ↓
                            After the repeated Perceptron blocks,
                      a final dropout and activation layer is applied:
                                           ↓
                               Dropout (p=dropout_rate)
                                           ↓
                               Activation (final_activation)

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

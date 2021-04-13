# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Applies a positional encoding to a tensor with shape (batch_size x seq_len x embed_dim).

    The positional encoding is computed as follows:
        PE(pos,2i) = sin(pos/10000^(2i/dmodel))
        PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

        where pos = position, pos in [0, seq_len)
        dmodel = data embedding dimension = embed_dim
        i = dimension index, i in [0, embed_dim)

    Reference: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
    Implementation Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embed_dim: int, seq_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim, dtype=torch.float)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pe.size(1) >= x.size(1), (
            "Cannot apply position encoding of size "
            + f"{self.pe.size()} when input has size {x.size()}"
        )
        return x + self.pe[:, : x.size(1), :]

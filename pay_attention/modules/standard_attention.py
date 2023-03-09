from __future__ import annotations

import math

import torch
from torch import Tensor

from ..utils import scaled, softmax
from ..utils import scaled_memory, softmax_memory


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    inplace: bool = False,
) -> Tensor:  # (B, T, C')
    """
    The standard function computes the attention mechanism between
    query, key, and value tensors using the standard approach. It
    scales the query tensor, calculates the attention scores between
    the query and key tensors, applies softmax activation on the scores,
    and computes the weighted sum of the value tensor based on the
    attention scores. The function returns the resulting attention output.
    """

    q = scaled(q, inplace)
    score = q @ k.transpose(-1, -2)  # (B, T, T')
    del q, k

    attn = softmax(score, inplace)  # (B, T, T')
    del score

    return attn @ v  # (B, T, C')


def standard_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    dtype: torch.dtype,
    inplace: bool = False,
) -> int:
    assert dtype in (torch.float32, torch.half)

    B, T, C = q_shape
    B, Tp, Cp = v_shape

    element_size = 4 if dtype == torch.float32 else 2

    size = scaled_memory(q_shape, dtype, inplace)
    size += (B * T * Tp) * element_size  # TODO matmul_memory
    size += softmax_memory((B, T, Tp), dtype, inplace)
    size += (B * T * Cp) * element_size

    # del q, k score; freed memory
    size -= (B * T * C) * element_size if not inplace else 0
    size -= (B * T * Tp) * element_size if not inplace else 0

    # Magic number
    size *= 1.239 if not inplace else 1

    return math.ceil(size)

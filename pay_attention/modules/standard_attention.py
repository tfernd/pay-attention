from __future__ import annotations

import math

import torch
from torch import Tensor

from ..ops import scaled, softmax
from ..ops import scaled_memory, softmax_memory


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    inplace: bool = False,
) -> Tensor:  # (B, T, C')
    """
    Computes the attention mechanism between query, key, and value tensors
    using the standard approach. The function scales the query tensor,
    calculates the attention scores between the query and key tensors,
    applies softmax activation on the scores, and computes the weighted
    sum of the value tensor based on the attention scores. The function
    returns the resulting attention output.
    """

    q = scaled(q, inplace)
    k = scaled(k, inplace)

    # Calculate the attention scores between the query and key tensors
    score = q @ k.transpose(-1, -2)  # (B, T, T')
    del q, k

    attn = softmax(score, inplace)  # (B, T, T')
    del score

    # Compute the weighted sum of the value tensor based on the attention scores
    return attn @ v  # (B, T, C')


def standard_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    dtype: torch.dtype,
    inplace: bool = False,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store the tensors
    used in the standard attention mechanism.
    """

    assert dtype in (torch.float32, torch.half)

    B, T, C = q_shape
    B, Tp, Cp = v_shape

    element_size = 4 if dtype == torch.float32 else 2

    size = scaled_memory(q_shape, dtype, inplace)
    size = scaled_memory(k_shape, dtype, inplace)
    size += (B * T * Tp) * element_size  # TODO matmul_memory
    size += softmax_memory((B, T, Tp), dtype, inplace)
    size += (B * T * Cp) * element_size  # TODO matmul_memory

    # TODO not perfect
    # Subtract the memory of the tensors that are no longer needed
    size -= (B * T * C) * element_size if not inplace else 0
    size -= (B * T * Tp) * element_size if not inplace else 0

    # Apply a magic number to account for non-power-of-two tensor (Empiric)
    size *= 1.239 if not inplace else 1

    return math.ceil(size)

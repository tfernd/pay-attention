from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ..utils import element_size, warp_memory_size, multiple
from ..ops import scaled, softmax
from ..ops import scaled_memory, softmax_memory
from ..ops import mask_score, mask_score_memory


def standard_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    mask: Optional[Tensor],  # (B?, T, T')
    inplace: bool,
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

    score = mask_score(score, mask, inplace)

    attn = softmax(score, inplace)  # (B, T, T')
    del score

    # Compute the weighted sum of the value tensor based on the attention scores
    return attn @ v  # (B, T, C')


# might fail in some very special edgy cases, rare
def standard_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    mask_shape: Optional[tuple[int, int] | tuple[int, int, int]],  # (B?, T, T')
    inplace: bool,
    dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
) -> int:
    """
    Computes the amount of memory (in bytes) required to store the tensors
    used in the standard attention mechanism.
    """

    B, T, C = q_shape
    B, Tp, Cp = v_shape
    score_shape = (B, T, Tp)

    warp_size = warp_memory_size(dtype)

    mem = scaled_memory(q_shape, inplace, dtype)
    mem += scaled_memory(k_shape, inplace, dtype)

    # matrix multiplication q @ k.T # TODO make a function for this!
    mem += element_size(dtype) * 2 * multiple(B * T * Tp, warp_size)

    mem += mask_score_memory(score_shape, mask_shape, inplace, dtype, mask_dtype)
    mem += softmax_memory(score_shape, inplace, dtype)

    # matrix multiplication attn @ v # TODO make a function for this!
    mem += element_size(dtype) * 2 * multiple(B * T * Cp, warp_size)

    return mem

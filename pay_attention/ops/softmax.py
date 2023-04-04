from __future__ import annotations

import math

import torch
from torch import Tensor


def softmax(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> Tensor:  # (..., C)
    """Applies the softmax function to the input tensor `x` along the last dimension."""

    if not inplace:
        return x.softmax(dim=-1, dtype=x.dtype)

    x -= x.amax(dim=-1, keepdim=True)
    x.exp_()
    x /= x.sum(dim=-1, keepdim=True)

    return x


# TODO computation does not work well for CPU and might be different for other GPUs
def softmax_memory(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the softmax function.
    """

    N = x.numel()
    C = x.size(-1)
    B = N // C

    element_size = 4 if x.dtype == torch.float32 else 2
    mult = 128 if x.dtype == torch.float32 else 256

    if inplace:
        B = math.ceil(B / mult + 1) * mult

        return element_size * B

    N = math.ceil(N / mult) * mult

    return element_size * 2 * N

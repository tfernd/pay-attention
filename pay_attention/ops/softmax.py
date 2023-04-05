from __future__ import annotations

import math
import numpy as np

import torch
from torch import Tensor


def softmax(
    x: Tensor,  # (...B, C)
    inplace: bool,
) -> Tensor:  # (...B, C)
    """Applies the softmax function to the input tensor `x` along the last dimension."""

    if not inplace:
        return x.softmax(dim=-1, dtype=x.dtype)

    x -= x.amax(dim=-1, keepdim=True)
    x.exp_()
    x /= x.sum(dim=-1, keepdim=True)

    return x


def softmax_memory(
    shape: tuple[int, ...],  # (...B, C)
    inplace: bool,
    dtype: torch.dtype,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the softmax function.
    """

    N = np.prod(shape)
    C = shape[-1]
    B = N // C

    element_size = 4 if dtype == torch.float32 else 2
    mult = 128 if dtype == torch.float32 else 256

    if inplace:
        B = math.ceil(B / mult + 1) * mult

        return element_size * B

    N = math.ceil(N / mult) * mult

    return element_size * 2 * N

from __future__ import annotations

import math
import numpy as np

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


def softmax_memory(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    inplace: bool = False,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the softmax function.
    """

    assert dtype in (torch.float32, torch.half)
    assert len(shape) >= 2

    *pre_shape, C = shape
    B = np.prod(pre_shape).item()

    element_size = 4 if dtype == torch.float32 else 2

    if inplace:
        # Compute the memory required to store the tensor in-place after applying the softmax function
        # by rounding up to the nearest multiple of the minimum size required to avoid bank conflicts
        # in shared memory
        min_size = 128 if dtype == torch.float32 else 256

        B += min_size - B % min_size

        return element_size * B

    # using a magic number that takes into account non-power of 2 B's and C's
    MAGIC = 1.0017

    return math.ceil(element_size * B * C * MAGIC)

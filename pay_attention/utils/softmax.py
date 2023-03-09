from __future__ import annotations

import math
import numpy as np

import torch
from torch import Tensor


def softmax(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> Tensor:# (..., C)
    """Applies the softmax function to the input tensor `x` along the last dimension."""

    if not inplace:
        return x.softmax(dim=-1, dtype=x.dtype)

    x -= x.amax(dim=-1, keepdim=True)
    x.exp_()
    x /= x.sum(dim=-1, keepdim=True)

    return x


# TODO code duplication
def softmax_memory(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> int:
    assert x.dtype in (torch.float32, torch.half)

    *shape, C = x.shape
    B = np.prod(shape).item()

    element_size = 4 if x.dtype == torch.float32 else 2

    if inplace:
        min_size = 128 if x.dtype == torch.float32 else 256 # cache size? warp-size?
        
        B += min_size - B % min_size

        return element_size * B

    # magic number that takes into account non-power of 2 B's ans C's
    MAGIC = 1.011

    return math.ceil(element_size * B * C * MAGIC)

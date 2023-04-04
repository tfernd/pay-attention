from __future__ import annotations

import math

import torch
from torch import Tensor


def scaled(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> Tensor:  # (..., C)
    """
    Scales the input tensor by multiplying it with the scaling factor.
    The scaling factor is calculated based on the dimension of the tensor (C)
    using the formula q / sqrt(C). The function returns the input tensor
    scaled by the calculated factor.
    """

    C = x.size(-1)
    scale = math.pow(C, -1 / 4)

    return x * scale if not inplace else x.mul_(scale)


def scaled_memory(
    x: Tensor,  # (..., C)
    inplace: bool = False,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the scaled function.
    """

    if inplace:
        return 0

    N = x.numel()

    element_size = 4 if x.dtype == torch.float32 else 2
    mult = 128 if x.dtype == torch.float32 else 256

    N = math.ceil(N / mult) * mult

    return element_size * N

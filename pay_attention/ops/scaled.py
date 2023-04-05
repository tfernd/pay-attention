from __future__ import annotations

import math
import numpy as np

import torch
from torch import Tensor


def scaled(
    x: Tensor,  # (...B, C)
    inplace: bool,
) -> Tensor:  # (...B, C)
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
    shape: tuple[int, ...],  # (...B, C)
    inplace: bool,
    dtype: torch.dtype,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the scaled function.
    """

    if inplace:
        return 0

    N = np.prod(shape)

    element_size = 4 if dtype == torch.float32 else 2
    mult = 128 if dtype == torch.float32 else 256

    N = math.ceil(N / mult) * mult

    return element_size * N

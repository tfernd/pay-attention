from __future__ import annotations

import math

import torch
from torch import Tensor

from ..utils import multiple, element_size, warp_memory_size


def scaled(
    x: Tensor,  # (...B?, C)
    inplace: bool,
) -> Tensor:  # (...B?, C)
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
    shape: tuple[int, ...],  # (...B?, C)
    inplace: bool,
    dtype: torch.dtype,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the scaled function.
    """

    if inplace:
        return 0

    N = math.prod(shape)
    warp_size = warp_memory_size(dtype)

    return element_size(dtype) * multiple(N, warp_size)

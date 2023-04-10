from __future__ import annotations

import math

import torch
from torch import Tensor

from ..utils import multiple, element_size, warp_memory_size


def softmax(
    x: Tensor,  # (...B?, C)
    inplace: bool,
) -> Tensor:  # (...B?, C)
    """Applies the softmax function to the input tensor `x` along the last dimension."""

    if not inplace:
        return x.softmax(dim=-1, dtype=x.dtype)

    xmax = x.amax(dim=-1, keepdim=True)
    x.sub_(xmax).exp_()

    xsum = x.sum(dim=-1, keepdim=True)
    x.div_(xsum)

    return x


def softmax_memory(
    shape: tuple[int, ...],  # (...B?, C)
    inplace: bool,
    dtype: torch.dtype,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store a tensor
    with the specified shape and data type after applying the softmax function.
    """

    N = math.prod(shape)
    C = shape[-1]
    B = N // C

    warp_size = warp_memory_size(dtype)

    if inplace:
        return element_size(dtype) * 2 * multiple(B, warp_size)  # xmax and xsum

    return element_size(dtype) * 2 * multiple(N, warp_size)  # F.softmax

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..utils import multiple


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

    # TODO create a function for this to avoid duplication
    element_size = 4 if dtype == torch.float32 else 2
    mult = 128 if dtype == torch.float32 else 256

    if inplace:
        return element_size * 2 * multiple(B, mult)  # xmax and xsum

    return element_size * 2 * multiple(N, mult)  # F.softmax

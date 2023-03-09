from __future__ import annotations

import math
import numpy as np

import torch
from torch import Tensor


def scaled(
    q: Tensor,  # (..., C)
    inplace: bool = False,
) -> Tensor:  # (..., C)
    """
    The scaled function scales the input tensor by multiplying
    it with the scaling factor. The scaling factor is calculated
    based on the dimension of the tensor (C) using the formula
    :math:`q / \\sqrt{C}`. The function returns
    the input tensor scaled by the calculated factor.
    """

    C = q.size(-1)
    scale = math.pow(C, -1 / 2)

    return q * scale if not inplace else q.mul_(scale)


# TODO code duplication
def scaled_memory(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    inplace: bool = False,
) -> int:
    if inplace:
        return 0

    assert dtype in (torch.float32, torch.half)
    assert len(shape) >= 2

    *pre_shape, C = shape
    B = np.prod(pre_shape).item()

    element_size = 4 if dtype == torch.float32 else 2

    # magic number that takes into account non-power of 2 B's ans C's
    MAGIC = 1.0035

    return math.ceil(element_size * B * C * MAGIC)

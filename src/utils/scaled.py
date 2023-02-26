from __future__ import annotations

import math

from torch import Tensor


def scaled(
    q: Tensor,  # (B, T, C)
) -> Tensor:
    """
    The scaled function scales the input tensor by multiplying
    it with the scaling factor. The scaling factor is calculated
    based on the dimension of the tensor (C) using the formula
    :math:`q / \\sqrt{C}`. The function returns
    the input tensor scaled by the calculated factor.
    """

    C = q.size(-1)
    scale = math.pow(C, -1 / 2)

    return q * scale

from __future__ import annotations

from typing import Optional
import numpy as np

import torch
from torch import Tensor


def mask_score(
    score: Tensor,  # (B...)
    mask: Optional[Tensor],  # (B...)
    inplace: bool,
) -> Tensor:  # (B...)
    if mask is None:
        return score

    if mask.dtype == torch.bool:
        mask = score.new_zeros(*mask.shape).masked_fill_(mask, float("-inf"))

    return score - mask if not inplace else score.sub_(mask)


def mask_score_memory(
    score_shape: tuple[int, ...],
    mask_shape: Optional[tuple[int, ...]],
    inplace: bool,
    score_dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
) -> int:
    assert (mask_shape is None) == (mask_dtype is None)

    if mask_shape is None:
        return 0

    N = np.prod(score_shape).item()
    element_size = 4 if score_dtype == torch.float32 else 2

    mem = element_size * N if not inplace else 0
    if mask_dtype == torch.bool:
        mem += 3 * element_size * N

    return mem

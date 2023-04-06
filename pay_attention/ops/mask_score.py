from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from ..utils import multiple


def mask_score(
    score: Tensor,  # (B, ...C)
    mask: Optional[Tensor],  # (B?, ...C)
    inplace: bool,
) -> Tensor:  # (B...)
    if mask is None:
        return score

    if mask.dtype == torch.bool:
        mask = score.new_zeros(*mask.shape).masked_fill_(mask, float("-inf"))

    return score + mask if not inplace else score.add_(mask)


def mask_score_memory(
    score_shape: tuple[int, ...],
    mask_shape: Optional[tuple[int, ...]],
    inplace: bool,
    score_dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
) -> int:
    if mask_shape is None:
        return 0

    Ns = math.prod(score_shape)
    Nm = math.prod(mask_shape) if mask_shape and mask_dtype == torch.bool is not None else 0
    element_size = 4 if score_dtype == torch.float32 else 2

    mem = element_size * Ns if not inplace else 0
    if mask_dtype == torch.bool:
        mem += 3 * element_size * Ns

    return mem

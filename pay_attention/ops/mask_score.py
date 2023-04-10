from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from ..utils import multiple, element_size, warp_memory_size

MIN = float("-inf")


def mask_score(
    score: Tensor,  # (B, ...C)
    mask: Optional[Tensor],  # (B?, ...C)
    inplace: bool,
) -> Tensor:  # (B...)
    """Masks the input tensor with a binary mask tensor."""

    if mask is None:
        return score

    if mask.dtype == torch.bool:
        if inplace:
            return score.masked_fill_(~mask, MIN)

        # TODO optimize this? needed?
        return score.masked_fill(~mask, MIN)

    return score + mask if not inplace else score.add_(mask)


def mask_score_memory(
    score_shape: tuple[int, ...],
    mask_shape: Optional[tuple[int, ...]],
    inplace: bool,
    score_dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
) -> int:
    """Returns the estimated memory usage of the mask_score operation."""

    if inplace or mask_shape is None:
        return 0

    Ns = math.prod(score_shape)
    warp_size = warp_memory_size(score_dtype)

    if mask_dtype is not None and mask_dtype == torch.bool:
        Nm = math.prod(mask_shape)

        # ? why 4? really 4? check!
        # There are some useles clone, empty-like, etc due to masked_fill
        return element_size(mask_dtype) * 4 * multiple(Nm, warp_size)

    return element_size(score_dtype) * multiple(Ns, warp_size)

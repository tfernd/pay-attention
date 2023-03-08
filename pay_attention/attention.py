from __future__ import annotations
from typing import Optional
from typing_extensions import Literal

import torch
from torch import Tensor

from .modules import standard
from .modules import batch_chunked, sequence_chunked


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    # mask: Tensor, #
) -> Tensor:
    # General attention computation

    # check shapes
    assert q.ndim == k.ndim == v.ndim == 3
    assert k.shape[:2] == v.shape[:2]
    assert q.size(0) == k.size(0)
    assert q.size(2) == k.size(2)

    return q  # ! TODO

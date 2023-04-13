from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

def standard_attention(
    query: Tensor,  # (B, T, C)
    key: Tensor,  # (B, T', C)
    value: Tensor,  # (B, T', C')
    mask: Optional[Tensor] = None,  # (B?, T, T')
) -> Tensor:  # (B, T, C')
    C = query.size(2)
    scale = math.pow(C, -1 / 4)

    query = query * scale
    key = key * scale

    attention_scores = query @ key.transpose(-1, -2)  # (B, T, T')

    if mask is not None:
        if mask.dtype == torch.bool:
            attention_scores.masked_fill_(~mask, float("-inf"))
        else:
            attention_scores += mask

    attention = attention_scores.softmax(dim=-1)  # (B, T, T')

    return attention @ value  # (B, T, C')

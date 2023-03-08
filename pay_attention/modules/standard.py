from __future__ import annotations

from torch import Tensor

from ..utils import scaled, softmax


def standard(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    *,
    inplace: bool = True,
) -> Tensor:  # (B, T, C')
    """
    The standard function computes the attention mechanism between
    query, key, and value tensors using the standard approach. It
    scales the query tensor, calculates the attention scores between
    the query and key tensors, applies softmax activation on the scores,
    and computes the weighted sum of the value tensor based on the
    attention scores. The function returns the resulting attention output.
    """

    q = scaled(q)
    score = q @ k.transpose(-1, -2)  # (B, T, T')
    del q, k

    attn = softmax(score, inplace=inplace)
    del score

    return attn @ v  # (B, T, C')

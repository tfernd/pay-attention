from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from .standard import standard


def batch_chunked(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    *,
    chunks: int,
    inplace: bool = True,
) -> Tensor:  # (B, T, C')
    """
    The batch_chunked function computes the attention mechanism
    between query, key, and value tensors using the batch-chunked
    approach. It iterates over the batches in chunks and computes the attention
    scores, applies softmax activation on the scores, and computes
    the weighted sum of the value tensor based on the attention
    scores.
    """

    assert chunks >= 1
    B, T, Cp = *q.shape[:2], v.size(2)

    out = torch.empty(B, T, Cp, dtype=q.dtype, device=q.device)  # (B, T, C')
    for i in range(0, B, chunks):
        torch.cuda.nvtx.range_push(f"batch-iter={i}")

        s = slice(i, min(i + chunks, B))

        out[s] = standard(q[s], k[s], v[s], inplace=inplace)  # (chunks, T, C')

        torch.cuda.nvtx.range_pop()

    return out


def sequence_chunked(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    *,
    chunks: int,
    inplace: bool = True,
) -> Tensor:  # (B, T, C')
    """
    The sequence_chunked function computes the attention mechanism
    between query, key, and value tensors using the sequence-chunked
    approach. It iterates over the sequence length in chunks and computes
    the attention scores, applies softmax activation on the scores,
    and computes the weighted sum of the value tensor based on the
    attention scores.
    """

    assert chunks >= 1
    B, T, Cp = *q.shape[:2], v.size(2)

    out = torch.empty(B, T, Cp, dtype=q.dtype, device=q.device)  # (B, T, C')
    for i in range(0, T, chunks):
        torch.cuda.nvtx.range_push(f"seq-iter={i}")

        s = slice(i, min(i + chunks, T))

        out[:, s] = standard(q[:, s], k, v, inplace=inplace)  # (B, chunks, C')

        torch.cuda.nvtx.range_pop()

    return out


def batch_and_sequence_chunked(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    *,
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
    inplace: bool = True,
) -> Tensor:  # (B, T, C')
    """
    Computes the attention mechanism between query, key,
    and value tensors using both the batch-chunked and sequence-chunked
    approaches. This function allows chunking over both the
    batch and sequence dimensions.
    """
    B, T, Cp = *q.shape[:2], v.size(2)

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    assert batch_chunks >= 1
    assert seq_chunks >= 1

    out = torch.empty(B, T, Cp, dtype=q.dtype, device=q.device)  # (B, T, C')

    for i in range(0, B, batch_chunks):
        torch.cuda.nvtx.range_push(f"batch-iter={i}")

        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            torch.cuda.nvtx.range_push(f"seq-iter={i}")

            sj = slice(j, min(j + seq_chunks, T))

            out[si, sj] = standard(q[si, sj], k[si], v[si], inplace=inplace)  # (batch-chunks, seq_chunks, C')

            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    return out

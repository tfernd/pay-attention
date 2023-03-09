from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from .standard_attention import standard_attention, standard_attention_memory


def batch_and_sequence_chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
    inplace: bool = False,
) -> Tensor:  # (B, T, C')
    """
    Computes the attention mechanism between query, key,
    and value tensors using both the batch-chunked and sequence-chunked
    approaches. This function allows chunking over both the
    batch and sequence dimensions.
    """

    B, T, C = q.shape
    B, Tp, Cp = v.shape

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    assert B >= batch_chunks >= 1
    assert T >= seq_chunks >= 1

    out = torch.empty(B, T, Cp, dtype=q.dtype, device=q.device)  # (B, T, C')
    for i in range(0, B, batch_chunks):
        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            sj = slice(j, min(j + seq_chunks, T))

            # (batch-chunks, seq_chunks, C')
            out[si, sj] = standard_attention(q[si, sj], k[si], v[si], inplace)

    return out


def batch_and_sequence_chunked_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    batch_chunks: int,
    seq_chunks: int,
    dtype: torch.dtype,
    inplace: bool = False,
) -> int:
    assert dtype in (torch.float32, torch.half)

    B, T, C = q_shape
    B, Tp, Cp = v_shape

    element_size = 4 if dtype == torch.float32 else 2

    q_shape = (batch_chunks, seq_chunks, C)
    v_shape = (batch_chunks, Tp, Cp)

    size = (B * T * Cp) * element_size
    size += standard_attention_memory(q_shape, v_shape, dtype, inplace)

    return size


def batch_chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    chunks: int,
    inplace: bool = False,
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

    return batch_and_sequence_chunked_attention(
        q, k, v, batch_chunks=chunks, seq_chunks=None, inplace=inplace
    )


def sequence_chunked_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    chunks: int,
    inplace: bool = False,
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

    return batch_and_sequence_chunked_attention(
        q, k, v, batch_chunks=None, seq_chunks=chunks, inplace=inplace
    )

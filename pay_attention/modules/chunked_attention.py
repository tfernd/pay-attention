from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from .standard_attention import standard_attention, standard_attention_memory


def chunked_attention(
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

    from tqdm.auto import trange as range

    out = torch.empty(B, T, Cp, dtype=q.dtype, device=q.device)  # (B, T, C')
    for i in range(0, B, batch_chunks):
        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            sj = slice(j, min(j + seq_chunks, T))

            # (batch-chunks, seq_chunks, C')
            out[si, sj] = standard_attention(q[si, sj], k[si], v[si], inplace)

    return out


def chunked_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    dtype: torch.dtype,
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
    inplace: bool = False,
) -> int:
    """
    Computes the amount of memory (in bytes) required to perform
    the batch-and-sequence chunked attention operation.
    """

    assert dtype in (torch.float32, torch.half)

    B, T, C = q_shape
    B, Tp, Cp = v_shape

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    assert B >= batch_chunks >= 1
    assert T >= seq_chunks >= 1
    
    element_size = 4 if dtype == torch.float32 else 2

    q_chunk_shape = (batch_chunks, seq_chunks, C)
    v_chunk_shape = (batch_chunks, Tp, Cp)

    size = (B * T * Cp) * element_size
    size += standard_attention_memory(q_chunk_shape, v_chunk_shape, dtype, inplace)

    return size

from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

from ..utils import available_memory
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

    # do not chunk it for nothing
    if batch_chunks == B and seq_chunks == T:
        return standard_attention(q, k, v, inplace)

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


def chunked_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
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

    if batch_chunks == B and seq_chunks == T:
        return standard_attention_memory(q_shape, k_shape, v_shape, dtype, inplace)

    assert B >= batch_chunks >= 1
    assert T >= seq_chunks >= 1

    q_chunk_shape = (batch_chunks, seq_chunks, C)
    k_chunk_shape = (batch_chunks, Tp, C)
    v_chunk_shape = (batch_chunks, Tp, Cp)

    element_size = 4 if dtype == torch.float32 else 2

    size = (B * T * Cp) * element_size  # cache size
    size += standard_attention_memory(q_chunk_shape, k_chunk_shape, v_chunk_shape, dtype, inplace)

    return size


def find_best_chunks(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int]:
    B, T, C = q_shape
    B, Tp, Cp = v_shape

    free_mem = available_memory(device)

    out: list[tuple[int, int, bool, int, float]] = []
    for inplace in [False, True]:
        for batch_chunks in range(1, B + 1):
            for i in range(1, 16):  # splits
                seq_chunks = T // i  # TODO use multiples of 128? 32?

                mem = chunked_attention_memory(
                    q_shape, k_shape, v_shape, dtype, batch_chunks, seq_chunks, inplace
                )
                if mem > free_mem:
                    continue

                loops = math.ceil(B / batch_chunks) * math.ceil(T / seq_chunks)

                if inplace:
                    # inplace make things a bit slower at the cost of less memory
                    loops += 1 / 2

                out.append((batch_chunks, seq_chunks, inplace, mem, loops))

    assert len(out) >= 1, "Potato PC went BOOM."
    out = sorted(out, key=lambda x: (x[3], x[4]))  # ? 4, 3?

    batch_chunks, seq_chunks, inplace, mem, loops = out[0]

    return batch_chunks, seq_chunks

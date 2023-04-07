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
    mask: Optional[Tensor],  # (B, T, T')
    inplace: bool,
    batch_chunks: Optional[int],
    seq_chunks: Optional[int],
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
        return standard_attention(q, k, v, mask, inplace)

    assert B >= batch_chunks >= 1
    assert T >= seq_chunks >= 1

    out = q.new_empty(B, T, Cp)  # (B, T, C')
    for i in range(0, B, batch_chunks):
        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            sj = slice(j, min(j + seq_chunks, T))

            # (batch-chunks?, seq_chunks, T')
            mask_chunk = None
            if mask is not None:
                mask_chunk = mask[sj] if mask.ndim == 2 else mask[si, sj]

            # (batch-chunks, seq_chunks, C')
            out[si, sj] = standard_attention(q[si, sj], k[si], v[si], mask_chunk, inplace)

    return out


def chunked_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    mask_shape: Optional[tuple[int, int, int]],  # (B, T, T')
    inplace: bool,
    batch_chunks: Optional[int],
    seq_chunks: Optional[int],
    dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
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
        return standard_attention_memory(q_shape, k_shape, v_shape, mask_shape, inplace, dtype, mask_dtype)

    assert B >= batch_chunks >= 1
    assert T >= seq_chunks >= 1

    q_chunk_shape = (batch_chunks, seq_chunks, C)
    k_chunk_shape = (batch_chunks, Tp, C)
    v_chunk_shape = (batch_chunks, Tp, Cp)
    mask_chunk_shape = (batch_chunks, seq_chunks, C)

    element_size = 4 if dtype == torch.float32 else 2

    mem = element_size * (B * T * Cp)  # cache size
    mem += standard_attention_memory(
        q_chunk_shape, k_chunk_shape, v_chunk_shape, mask_chunk_shape, inplace, dtype, mask_dtype
    )

    return mem


def find_best_chunks(
    q_shape: tuple[int, int, int],  # (B, T, C)
    k_shape: tuple[int, int, int],  # (B, T', C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    mask_shape: Optional[tuple[int, int, int]],  # (B, T, T')
    dtype: torch.dtype,
    mask_dtype: Optional[torch.dtype],
    device: torch.device,
) -> tuple[int, int, bool, float]:
    B, T, C = q_shape
    B, Tp, Cp = v_shape

    free_mem = available_memory(device)

    out: list[tuple[float, int, int, int, bool]] = []
    for inplace in [False, True]:
        for batch_chunks in range(B, 0, -1):
            for i in range(1, T):  # splits
                seq_chunks = T // i  # TODO use multiples of 128? 32? better?

                mem = chunked_attention_memory(
                    q_shape,
                    k_shape,
                    v_shape,
                    mask_shape,
                    inplace,
                    batch_chunks,
                    seq_chunks,
                    dtype,
                    mask_dtype,
                )

                if mem > free_mem:
                    continue

                loops = math.ceil(B / batch_chunks) * math.ceil(T / seq_chunks)

                if inplace:
                    # inplace make things a bit slower at the cost of less memory
                    # so we pretend it takes half more iterations
                    loops += 1 / 2

                out.append((loops, mem, batch_chunks, seq_chunks, inplace))

                # alread found the biggest that fit, no need to divide T any longer
                break

    assert len(out) >= 1, "Potato PC went BOOM!"

    # least amount of loops and memory
    out = sorted(out, key=lambda x: (x[0], x[1]))

    loops, mem, batch_chunks, seq_chunks, inplace = out[0]

    return batch_chunks, seq_chunks, inplace, loops

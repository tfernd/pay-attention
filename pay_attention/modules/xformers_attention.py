from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor

try:
    import xformers.ops

    XFORMERS = True
except:
    XFORMERS = False

from ..utils import available_memory


def xformers_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
) -> Tensor:  # (B, T, C')
    """
    Computes the attention mechanism between query, key, and value tensors
    using the XFormers approach. The function converts the query, key,
    and value tensors to half-precision floating-point format, applies the
    XFormers memory-efficient attention operation, and returns the resulting
    attention output in the original data type of the query tensor.
    """

    assert XFORMERS
    op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp

    B, T, C = q.shape
    B, Tp, Cp = v.shape

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    dtype = q.dtype
    if dtype != torch.half:
        q = q.half()
        k = k.half()
        v = v.half()

    # do not chunk it for nothing
    if batch_chunks == B and seq_chunks == T:
        out = xformers.ops.memory_efficient_attention(q, k, v, op=op)

        return out.to(dtype)

    out = torch.empty(B, T, Cp, dtype=dtype, device=q.device)  # (B, T, C')
    for i in range(0, B, batch_chunks):
        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            sj = slice(j, min(j + seq_chunks, T))

            # (batch-chunks, seq_chunks, C')
            out[si, sj] = xformers.ops.memory_efficient_attention(q[si, sj], k[si], v[si], op=op).to(dtype)

    return out


def xformers_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C)
    dtype: torch.dtype,
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
) -> int:
    """
    Computes the amount of memory (in bytes) required to store the tensors
    used in the XFormers attention mechanism. The function returns the
    estimated memory usage based on the input tensor shapes.
    """

    B, T, C = q_shape
    B, Tp, C = v_shape

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    size = batch_chunks * C * (8 * seq_chunks + 4 * Tp)

    if batch_chunks != B or seq_chunks != T:
        element_size = 4 if dtype == torch.float32 else 2
        size += (B * T * C) * element_size  # cache size

    return size


def find_xformers_best_chunks(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C')
    dtype: torch.dtype,
    device: torch.device,  # ! CUDA?
) -> tuple[int, int]:
    B, T, C = q_shape
    B, Tp, Cp = v_shape

    free_mem = available_memory(device=device)  # TODO M1 devices?

    out: list[tuple[int, int, int, int]] = []
    for batch_chunks in range(1, B + 1):
        for i in range(1, 16):  # splits
            seq_chunks = T // i  # TODO use multiples of 128? 32?

            mem = xformers_attention_memory(q_shape, v_shape, dtype, batch_chunks, seq_chunks)
            if mem > free_mem:
                continue

            loops = math.ceil(B / batch_chunks) * math.ceil(T / seq_chunks)

            out.append((batch_chunks, seq_chunks, mem, loops))

    # TODO sort by least loops and least memory

    assert len(out) >= 1, "Potato PC went BOOM."
    out = sorted(out, key=lambda x: (x[2], x[3]))  # ? 3, 2?

    batch_chunks, seq_chunks, mem, loops = out[0]

    return batch_chunks, seq_chunks

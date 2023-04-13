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
    /,
    *,
    batch_chunks: Optional[int] = None,
    seq_chunks: Optional[int] = None,
) -> Tensor:  # (B, T, C')
    B, T, C = query.shape
    B, Tp, Cp = value.shape

    # default
    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    is_chunked = batch_chunks != B or seq_chunks != T

    assert 1 <= batch_chunks <= B
    assert 1 <= seq_chunks <= T

    # scale query and key
    scale = math.pow(C, -1 / 4)
    query = query * scale
    key = key * scale

    # temporary array in case of chunked computation
    # (B, T, C')
    out = query.new_empty(B, T, Cp) if is_chunked else None

    for i in range(0, B, batch_chunks):
        batch_slice = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            seq_slice = slice(j, min(j + seq_chunks, T))

            # (batch-chunks, seq_chunks, T')
            scores = query[batch_slice, seq_slice] @ key[batch_slice].transpose(-1, -2)

            if mask is not None:
                # (batch-chunks?, seq_chunks, T')
                mask_chunk = mask[seq_slice] if mask.ndim == 2 else mask[batch_slice, seq_slice]

                # in-place
                if mask_chunk.dtype == torch.bool:
                    scores.masked_fill_(~mask_chunk, float("-inf"))
                else:
                    scores += mask_chunk

            # (batch-chunks, seq_chunks, T')
            attention = scores.softmax(dim=-1)

            # (batch-chunks, seq_chunks, T')
            out_chunk = attention @ value[batch_slice]

            if out is not None:
                out[batch_slice, seq_slice] = out_chunk
            else:
                return out_chunk

    assert out is not None

    return out

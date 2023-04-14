from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor


def standard_attention(
    query: Tensor,  # shape(B, T, C)
    key: Tensor,  # shape(B, T', C)
    value: Tensor,  # shape(B, T', C')
    mask: Optional[Tensor] = None,  # shape(B,? T, T')
    /,
    *,
    num_batch_chunks: int = 1,
    num_seq_chunks: int = 1,
) -> Tensor:  # shape(B, T, C')
    B, T, C = query.shape
    B, Tp, Cp = value.shape

    # scale query and key
    scale = math.pow(C, -1 / 4)
    query = query * scale
    key = key * scale

    batch_chunk_size = math.ceil(B / num_batch_chunks)
    seq_chunk_size = math.ceil(T / num_seq_chunks)

    out = query.new_empty(B, T, Cp)  # (B, T, C')
    for batch_idx in range(0, B, batch_chunk_size):
        batch_slice = slice(batch_idx, min(batch_idx + batch_chunk_size, B))

        for seq_idx in range(0, T, seq_chunk_size):
            seq_slice = slice(seq_idx, min(seq_idx + seq_chunk_size, T))

            # (batch-chunks, seq_chunks, T')
            scores = query[batch_slice, seq_slice] @ key[batch_slice].transpose(-1, -2)

            if mask is not None:
                # (batch-chunks?, seq_chunks, T')
                mask_chunk = mask[..., seq_slice, :]

                # in-place
                if mask_chunk.dtype == torch.bool:
                    scores.masked_fill_(~mask_chunk, float("-inf"))
                else:
                    scores += mask_chunk

            # (batch-chunks, seq_chunks, T')
            # attention = scores.softmax(dim=-1)
            scores -= scores.amax(dim=-1, keepdim=True)
            attention = scores.exp_()
            attention /= attention.sum(dim=-1, keepdim=True)

            # (batch-chunks, seq_chunks, T')
            torch.bmm(attention, value[batch_slice], out=out[batch_slice, seq_slice])

    return out

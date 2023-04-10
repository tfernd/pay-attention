from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

try:
    from torch.nn.functional import scaled_dot_product_attention

    TORCH2 = True
except:
    TORCH2 = False


def chunked_scaled_dot_product_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    mask: Optional[Tensor],  # (B?, T, T')
    batch_chunks: Optional[int],
    seq_chunks: Optional[int],
) -> Tensor:  # (B, T, C')
    assert TORCH2

    B, T, C = q.shape
    B, Tp, Cp = v.shape

    batch_chunks = batch_chunks or B
    seq_chunks = seq_chunks or T

    dtype = q.dtype

    # do not chunk it for nothing
    if batch_chunks == B and seq_chunks == T:
        return scaled_dot_product_attention(q, k, v, mask)

    out = torch.empty(B, T, Cp, dtype=dtype, device=q.device)  # (B, T, C')
    for i in range(0, B, batch_chunks):
        si = slice(i, min(i + batch_chunks, B))

        for j in range(0, T, seq_chunks):
            sj = slice(j, min(j + seq_chunks, T))

            # (batch-chunks?, seq_chunks, T')
            mask_chunk = None
            if mask is not None:
                mask_chunk = mask[sj] if mask.ndim == 2 else mask[si, sj]

            # (batch-chunks, seq_chunks, C')
            out[si, sj] = scaled_dot_product_attention(q[si, sj], k[si], v[si], mask_chunk).to(dtype)

    return out

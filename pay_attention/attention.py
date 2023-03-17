from __future__ import annotations

from torch import Tensor

from .modules import chunked_attention, find_best_chunks, find_xformers_best_chunks
from .modules import xformers_attention, XFORMERS


# TODO add torch 2 attention
def attention(
    q: Tensor, k: Tensor, v: Tensor, inplace: bool = False  # (B, T, C)  # (B, T', C)  # (B, T', C')
) -> Tensor:  # (B, T, C')
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.size(0) == k.size(0) == v.size(0)  # B
    assert q.size(2) == k.size(2)  # C
    assert k.size(1) == v.size(1)  # T'

    B, T, C = q.shape
    B, Tp, Cp = k.shape

    q, k, v = map(lambda x: x.contiguous(), (q, k, v))  # ? needed?

    # TODO check if is CUDA device?
    if XFORMERS and C == Cp and C <= 128:
        batch_chunks, seq_chunks = find_xformers_best_chunks(q.shape, v.shape, q.device)

        return xformers_attention(q, k, v, batch_chunks, seq_chunks)

    batch_chunks, seq_chunks = find_best_chunks(q.shape, k.shape, v.shape, q.dtype, q.device)

    return chunked_attention(q, k, v, batch_chunks, seq_chunks, inplace)

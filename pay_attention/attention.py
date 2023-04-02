from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .modules import chunked_attention, find_best_chunks, find_xformers_best_chunks
from .modules import xformers_attention, XFORMERS
from .modules import memory_efficient_attention


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    inplace: bool = False,
) -> Tensor:  # (B, T, C')
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.size(0) == k.size(0) == v.size(0)  # B
    assert q.size(2) == k.size(2)  # C
    assert k.size(1) == v.size(1)  # T'

    device = q.device
    B, T, C = q.shape
    B, Tp, Cp = k.shape

    q, k, v = map(lambda x: x.contiguous(), (q, k, v))  # ? needed?

    # TODO add torch 2 attention
    # return F.scaled_dot_product_attention(q, k, v)

    if XFORMERS and C == Cp and C <= 128 and device.type == 'cuda':
        batch_chunks, seq_chunks = find_xformers_best_chunks(q.shape, v.shape, q.dtype, q.device)

        return xformers_attention(q, k, v, batch_chunks, seq_chunks)

    batch_chunks, seq_chunks = find_best_chunks(q.shape, k.shape, v.shape, q.dtype, q.device)

    return chunked_attention(q, k, v, batch_chunks, seq_chunks, inplace)

    # memory_efficient_attention(q, k, v)

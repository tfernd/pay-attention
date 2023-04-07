from __future__ import annotations
from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from .modules import XFORMERS, xformers_attention, find_xformers_best_chunks
from .modules import chunked_attention, find_best_chunks

# from .modules import memory_efficient_attention


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    mask: Optional[Tensor] = None,  # (B?, T, T')
) -> Tensor:  # (B, T, C')
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.size(0) == k.size(0) == v.size(0)  # B
    assert q.size(2) == k.size(2)  # C
    assert k.size(1) == v.size(1)  # T'

    if mask is not None:
        assert mask.ndim == (2, 3)
        assert mask.size(-2) == q.size(1)  # T
        assert mask.size(-1) == k.size(1)  # T'

        mask = mask.contiguous()  # ? needed?

    device = q.device
    B, T, C = q.shape
    B, Tp, Cp = k.shape

    q, k, v = map(lambda x: x.contiguous(), (q, k, v))  # ? needed?

    # add batch and sequence chunks
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, mask)

    if XFORMERS and C == Cp and C <= 128 and device.type == "cuda" and mask is None:  # TODO use mask
        batch_chunks, seq_chunks = find_xformers_best_chunks(q.shape, v.shape, q.dtype, q.device)

        return xformers_attention(q, k, v, mask, batch_chunks, seq_chunks)

    mask_shape = mask.shape if mask is not None else None
    mask_dtype = mask.dtype if mask is not None else None
    batch_chunks, seq_chunks, inplace, loops = find_best_chunks(
        q.shape, k.shape, v.shape, mask_shape, q.dtype, mask_dtype, q.device
    )

    return chunked_attention(q, k, v, mask, inplace, batch_chunks, seq_chunks)

    # ! memory_efficient_attention(q, k, v)

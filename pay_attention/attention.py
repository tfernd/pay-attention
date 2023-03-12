from __future__ import annotations

from torch import Tensor

from .utils import available_memory
from .modules import standard_attention, standard_attention_memory
from .modules import chunked_attention, chunked_attention_memory
from .modules import xformers_attention, XFORMERS


def attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
) -> Tensor:  # (B, T, C')
    B, T, C = q.shape
    B, Tp, Cp = k.shape

    assert q.ndim == k.ndim == v.ndim == 3
    assert q.size(0) == k.size(0) == v.size(0)  # B
    assert q.size(2) == k.size(2)  # C
    assert k.size(1) == v.size(1)  # T'

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # TODO check if is CUDA device
    if XFORMERS and C == Cp and C <= 128:
        return xformers_attention(q, k, v)

    free_mem = available_memory(device=q.device)  # TODO M1 devices?

    # Try standard attention
    for inplace in [False, True]:
        mem = standard_attention_memory(q.shape, v.shape, q.dtype, inplace)

        if mem < free_mem:
            return standard_attention(q, k, v, inplace)

    # Try all possible batch and sequency combinations and get the one that uses the most RAM < free
    out: list[tuple[int, int, bool, int]] = []
    for inplace in [False, True]:
        for batch_chunks in range(1, B + 1):
            for i in range(1, 16):  # splits
                seq_chunks = T // i  # TODO use multiples of 128? 32?

                mem = chunked_attention_memory(q.shape, v.shape, q.dtype, batch_chunks, seq_chunks, inplace)
                if mem > free_mem:
                    continue

                out.append((batch_chunks, seq_chunks, inplace, mem))
    assert len(out) >= 1
    out = sorted(out, key=lambda x: x[-1], reverse=True)

    batch_chunks, seq_chunks, inplace, _ = out[0]

    return chunked_attention(q, k, v, batch_chunks, seq_chunks, inplace)

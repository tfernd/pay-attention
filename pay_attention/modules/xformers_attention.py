from __future__ import annotations

from torch import Tensor

try:
    import xformers.ops

    XFORMERS = True
except:
    XFORMERS = False


def xformers_attention(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
) -> Tensor:  # (B, T, C')
    assert XFORMERS

    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()

    op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
    out = xformers.ops.memory_efficient_attention(q, k, v, op=op)

    return out.to(dtype)


def xformers_attention_memory(
    q_shape: tuple[int, int, int],  # (B, T, C)
    v_shape: tuple[int, int, int],  # (B, T', C)
) -> int:
    B, T, C = q_shape
    B, Tp, C = v_shape

    return B * C * (8 * T + 4 * Tp)

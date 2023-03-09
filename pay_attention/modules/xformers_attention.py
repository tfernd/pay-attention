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
    op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=op)

    return out

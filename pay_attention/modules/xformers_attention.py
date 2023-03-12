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
    """
    Computes the attention mechanism between query, key, and value tensors
    using the XFormers approach. The function converts the query, key,
    and value tensors to half-precision floating-point format, applies the
    XFormers memory-efficient attention operation, and returns the resulting
    attention output in the original data type of the query tensor.
    """

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
    """
    Computes the amount of memory (in bytes) required to store the tensors
    used in the XFormers attention mechanism. The function returns the
    estimated memory usage based on the input tensor shapes.
    """

    B, T, C = q_shape
    B, Tp, C = v_shape

    return B * C * (8 * T + 4 * Tp)

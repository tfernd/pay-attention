from __future__ import annotations

from torch import Tensor


def softmax(
    x: Tensor,
    *,
    inplace: bool,
) -> Tensor:
    # TODO doc-string

    if not inplace:
        return x.softmax(dim=-1, dtype=x.dtype)

    x -= x.amax(dim=-1, keepdim=True)
    x.exp_()
    x /= x.sum(dim=-1, keepdim=True)

    return x

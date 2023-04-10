from __future__ import annotations

import torch


def element_size(dtype: torch.dtype, /) -> int:
    """Returns the size of an element in bytes for a given data type."""

    assert dtype in (torch.float32, torch.half)

    return 4 if dtype == torch.float32 else 2

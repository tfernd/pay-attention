from __future__ import annotations

import torch


def warp_memory_size(dtype: torch.dtype, /) -> int:
    """Returns the memory alignment requirement for optimal memory access on CUDA devices."""

    assert dtype in (torch.float32, torch.half)

    return 128 if dtype == torch.float32 else 256

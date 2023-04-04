from __future__ import annotations
from typing import Optional

import gc
import torch


def clear_cuda_mem(device: torch.device) -> None:
    """Clears the memory of the specified CUDA device."""

    torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def clear_mem(device: Optional[torch.device] = None) -> None:
    """Clears CUDA/CPU-virtual memory and performs garbage collection."""

    gc.collect()

    if device is None:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                clear_cuda_mem(device)

    elif device.type == "cuda":
        assert torch.cuda.is_available()
        clear_cuda_mem(device)

    # call gc.collect() again after freeing memory to free any objects that were holding memory
    gc.collect()

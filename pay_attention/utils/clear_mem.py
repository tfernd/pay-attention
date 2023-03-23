from __future__ import annotations

import gc
import torch


def clear_mem(device: torch.device) -> None:
    """Clears CUDA/CPU-virtual memory and performs garbage collection."""

    gc.collect()

    if device.type == "cuda":
        assert torch.cuda.is_available()

        torch.cuda.synchronize(device)  # ? needed?

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # call gc.collect() again after freeing memory to free any objects that were holding memory
    gc.collect()

from __future__ import annotations
from typing import Optional

import gc
import torch


def clear_cuda(sync_device: Optional[torch.device] = None) -> None:
    """Clear CUDA memory and garbage collection."""

    gc.collect()
    
    if torch.cuda.is_available():
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

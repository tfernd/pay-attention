from __future__ import annotations
from typing import Optional

import torch


def available_memory(device: Optional[torch.device] = None) -> int:
    """
    Returns the amount of available memory in bytes on the
    specified `device`, or the default device if `device` is None.
    """

    assert torch.cuda.is_available()
    # TODO M1/CPU

    stats = torch.cuda.memory_stats(device)

    reserved = stats["reserved_bytes.all.current"]
    active = stats["active_bytes.all.current"]
    free, total = torch.cuda.mem_get_info(device)

    free += reserved - active

    return free


# TODO ADD
# psutil.virtual_memory().available


def allocated_memory(device: Optional[torch.device] = None) -> int:
    """
    Returns the current amount of allocated memory in bytes
    on the specified `device`, or the default device if `device` is None.
    This includes memory that has been reserved for the current
    allocation but not yet used.
    """

    # TODO M1/CPU
    memory_stats = torch.cuda.memory_stats(device)
    allocated_memory = memory_stats["allocated_bytes.all.current"]

    return allocated_memory

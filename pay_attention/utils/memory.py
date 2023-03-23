from __future__ import annotations

import psutil

import torch


def available_memory(device: torch.device) -> int:
    """
    Returns the amount of available memory in bytes on the
    specified `device`, or the default device if `device` is None.
    """

    if device.type == "cuda":
        assert torch.cuda.is_available()

        stats = torch.cuda.memory_stats(device)

        reserved = stats["reserved_bytes.all.current"]
        active = stats["active_bytes.all.current"]
        free, total = torch.cuda.mem_get_info(device)

        free += reserved - active

        return free

    # CPU
    return psutil.virtual_memory().available


def allocated_memory(device: torch.device) -> int:
    """
    Returns the current amount of allocated memory in bytes
    on the specified `device`, or the default device if `device` is None.
    This includes memory that has been reserved for the current
    allocation but not yet used.
    """

    if device.type == "cuda":
        assert torch.cuda.is_available()

        memory_stats = torch.cuda.memory_stats(device)
        allocated_memory = memory_stats["allocated_bytes.all.current"]

        return allocated_memory

    # CPU
    # Get the memory usage of the current process
    process = psutil.Process()
    mem_info = process.memory_info()

    return mem_info.rss  # return resident set size

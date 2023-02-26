from __future__ import annotations
from typing import Callable, Any, Optional

import threading
import torch

from .clear_cuda import clear_cuda
from .memory import allocated_memory


def memory_used_by_function(
    function: Callable[..., Any],
    *args: Any,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> int:
    """
    Measures the GPU memory allocated during the execution of a function.
    """

    clear_cuda(device)

    allocated_list: list[int] = []

    # Set up a thread to measure allocated memory usage.
    exit_flag = threading.Event()
    thread = threading.Thread(target=pool_allocated_memory_continuous, args=(allocated_list, exit_flag))
    thread.start()

    initial_allocated_memory = allocated_memory()

    # Execute the provided function.
    function(*args, **kwargs)

    # Signal the memory usage measuring thread to exit and wait for it to join.
    exit_flag.set()
    thread.join()

    allocated_memory_used = max(allocated_list) - initial_allocated_memory if len(allocated_list) >= 1 else 0

    clear_cuda(device)

    return allocated_memory_used


def pool_allocated_memory_continuous(
    allocated_list: list[int],
    exit_flag: threading.Event,
) -> None:
    """
    Measures the GPU memory allocated during the execution of a function.
    """

    # Continuously measure the allocated memory while the exit flag is not set.
    while not exit_flag.is_set():
        allocated_list.append(allocated_memory())

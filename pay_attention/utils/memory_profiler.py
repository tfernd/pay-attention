from __future__ import annotations

from torch.profiler import profile

from .clear_mem import clear_mem


class MemoryProfiler:
    cuda_mem: int = 0
    cpu_mem: int = 0

    def __init__(self, show: bool = False, /) -> None:
        self.show = show

    def __enter__(self):
        clear_mem()

        self.profiler = profile(profile_memory=True)
        self.profiler.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.__exit__(exc_type, exc_value, traceback)

        for k in self.profiler.key_averages():
            if k.cuda_memory_usage > 0:
                self.cuda_mem += k.cuda_memory_usage / k.count
            if k.cpu_memory_usage > 0:
                self.cpu_mem += k.cpu_memory_usage / k.count

        if self.show:
            print(self.profiler.key_averages())

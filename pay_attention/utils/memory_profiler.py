from __future__ import annotations

from torch.profiler import profile


# It will be slightly off in case of CPU memory, dunno why...
# In CUDA mode sometimes it will give WAY more memory than what is actually used
# One trick Im doing to fix this is to run the function a few times and get the lowers mem usage... But it fell bad to do this..
# Is there a way to FIX this?!!? I have to manual re-run things to check... So lame... Better profiling?
class MemoryProfiler:
    cuda_mem: int
    cpu_mem: int

    def __init__(self, show: bool = False, /) -> None:
        self.show = show

    def __enter__(self):
        self.profiler = profile(profile_memory=True)
        self.profiler.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.__exit__(exc_type, exc_value, traceback)

        self.cuda_mem = self.cpu_mem = 0
        for k in self.profiler.key_averages():
            if k.cuda_memory_usage > 0:
                self.cuda_mem += k.cuda_memory_usage
            if k.cpu_memory_usage > 0:
                self.cpu_mem += k.cpu_memory_usage

        if self.show:
            print(self.profiler.key_averages())

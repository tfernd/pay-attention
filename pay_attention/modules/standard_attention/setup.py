from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="standard_attention_cuda",
    ext_modules=[
        CUDAExtension(
            "standard_attention_cuda",
            sources=[
                "standard_attention_wrapper.cpp",
                "standard_attention_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

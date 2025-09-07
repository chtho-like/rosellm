"""
Minimal setup.py for handling CUDA extensions.
All other configuration is in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup

# Check if CUDA extension files exist
cuda_ext = []
cmdclass = {}

# Check for CUDA extension source files
cuda_cpp = Path("flash_attention_cuda.cpp")
cuda_cu = Path("flash_attention_cuda.cu")

if cuda_cpp.exists() and cuda_cu.exists():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        cuda_ext = [
            CUDAExtension(
                name="rosellm.flash_attention_cuda",
                sources=["flash_attention_cuda.cpp", "flash_attention_cuda.cu"],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
        print("CUDA extension files found - will build flash_attention_cuda")
    except ImportError:
        print("Warning: torch not available, skipping CUDA extension")
else:
    print("Note: CUDA extension files not found, building pure Python package")

# Minimal setup call - everything else comes from pyproject.toml
setup(
    ext_modules=cuda_ext,
    cmdclass=cmdclass,
)

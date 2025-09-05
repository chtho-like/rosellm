from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rosellm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.20.0",
    ],
    author="wineandchord",
    author_email="guoqizhou123123@gmail.com",
    description="A comprehensive RLHF framework written from scratch",
    keywords="llm, rlhf, distributed training",
    python_requires=">=3.8",
    ext_modules=[
        CUDAExtension(
            name="flash_attention_cuda",
            sources=["flash_attention_cuda.cpp", "flash_attention_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

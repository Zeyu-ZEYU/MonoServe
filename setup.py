"""Build the MonoServe native library (monoserve._C).

The CUDA sources need CUTLASS and ThunderKittens under third_party/;
scripts/fetch_deps.sh places the pinned versions there.
"""
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
TP = ROOT / "third_party"

sources = [
    "csrc/bindings.cpp",
    "csrc/host/fabric.cpp",
    "csrc/host/tensor_map.cpp",
    "csrc/host/host_loop.cpp",
    "csrc/host/green.cpp",
    "csrc/fabric/fabric_kernel.cu",
    "csrc/control/estimator.cpp",
    "csrc/control/search.cpp",
    "csrc/control/admission.cpp",
    "csrc/control/py_control.cpp",
]
include_dirs = [
    str(ROOT / "csrc"),
    str(TP / "cutlass" / "include"),
    str(TP / "cutlass" / "tools" / "util" / "include"),
    str(TP / "ThunderKittens" / "include"),
]
nvcc_flags = [
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_90a,code=sm_90a",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DKITTENS_HOPPER",
    "-DNDEBUG",
    "-lineinfo",
    "-Xcompiler=-fno-strict-aliasing",
]
if os.environ.get("MONOSERVE_PTXAS_VERBOSE"):
    nvcc_flags.append("-Xptxas=-v")

setup(
    name="monoserve",
    version="0.1.0",
    description="MoE serving with CPU-resident experts and a contention-gated "
                "kernel fabric",
    packages=find_packages(include=["monoserve", "monoserve.*"]),
    ext_modules=[
        CUDAExtension(
            name="monoserve._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O3", "-std=c++20"],
                                "nvcc": nvcc_flags},
            libraries=["cuda"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    # the Local baseline's expert placement (which MuxWise runs on too)
    # hooks into vLLM as a general plugin, active only when vLLM's
    # additional config asks for it
    entry_points={"vllm.general_plugins": [
        "monoserve_baselines = monoserve.baselines:register"]},
)

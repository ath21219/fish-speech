"""
CUDA kernels for GGUF quantized GEMV.

Provides dp4a-based fused GEMV kernels derived from llama.cpp,
replacing the Triton kernels for better performance on Turing+ GPUs.

Build: JIT-compiled on first import via torch.utils.cpp_extension.load().
       No setup.py required. Compilation results are cached in:
         ~/.cache/torch_extensions/ (Linux)
         %LOCALAPPDATA%\\torch_extensions\\ (Windows)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

_CUDA_KERNELS_AVAILABLE = False
_cuda_ext = None


def _add_dll_directories():
    """Add CUDA and PyTorch lib directories to the DLL search path on Windows."""
    try:
        import torch
        # PyTorch ships its own CUDA runtime DLLs
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))

        # Also add the CUDA toolkit bin directory if available
        for env_var in ("CUDA_PATH", "CUDA_PATH_V12_9", "CUDA_PATH_V13_0", "CUDA_PATH_V13_2"):
            cuda_path = os.environ.get(env_var)
            if cuda_path:
                cuda_bin = Path(cuda_path) / "bin"
                if cuda_bin.is_dir():
                    os.add_dll_directory(str(cuda_bin))
                break
    except Exception:
        pass  # best-effort


def _try_load():
    """Attempt to load pre-built extension, then fall back to JIT compile."""
    global _CUDA_KERNELS_AVAILABLE, _cuda_ext

    # ── 1. Try pre-built (if installed via setup.py/pip at some point) ──
    try:
        from . import _gguf_cuda_ext as ext  # noqa: F811
        _cuda_ext = ext
        _CUDA_KERNELS_AVAILABLE = True
        logger.info("Loaded pre-built CUDA GGUF extension")
        return
    except ImportError:
        pass

    # ── 2. JIT compile ──
    try:
        import torch
        if not torch.cuda.is_available():
            logger.debug("CUDA not available, skipping CUDA GGUF kernels")
            return

        # On Windows, ensure CUDA and PyTorch DLLs are discoverable
        if sys.platform == "win32":
            _add_dll_directories()

        from torch.utils.cpp_extension import load

        csrc_dir = Path(__file__).parent / "csrc"
        sources = [
            str(csrc_dir / "bindings.cpp"),
            str(csrc_dir / "gguf_gemv.cu"),
        ]

        # Verify source files exist
        for src in sources:
            if not Path(src).exists():
                logger.warning(f"CUDA kernel source not found: {src}")
                return

        # Determine compute capabilities
        cc = torch.cuda.get_device_capability(0)
        sm = f"{cc[0]}{cc[1]}"
        gencode = f"-gencode=arch=compute_{sm},code=sm_{sm}"

        # Common nvcc flags
        nvcc_flags = [
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            gencode,
        ]

        # On Windows, avoid -std=c++17 for nvcc host compiler (MSVC doesn't need it)
        cxx_flags = ["/O2"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

        logger.info(
            f"JIT compiling CUDA GGUF kernels (sm_{sm})... "
            f"(first run only, cached afterwards)"
        )

        ext = load(
            name="_gguf_cuda_ext",
            sources=sources,
            extra_cuda_cflags=nvcc_flags,
            extra_cflags=cxx_flags,
            verbose=os.environ.get("GGUF_CUDA_VERBOSE", "") == "1",
        )

        _cuda_ext = ext
        _CUDA_KERNELS_AVAILABLE = True
        logger.info(f"CUDA GGUF kernels compiled and loaded (sm_{sm})")

    except Exception as e:
        logger.warning(
            f"Failed to JIT compile CUDA GGUF kernels: {e}. "
            f"Falling back to Triton/PyTorch."
        )
        if os.environ.get("GGUF_CUDA_VERBOSE", "") == "1":
            import traceback
            traceback.print_exc()


# Run at import time
_try_load()


def fused_gemv_q6k(input_vec, q_weight_data, D_in, D_out, dtype=None):
    """Q6_K fused GEMV using dp4a CUDA kernel."""
    if not _CUDA_KERNELS_AVAILABLE:
        raise RuntimeError("CUDA GGUF kernels not available")
    out = _cuda_ext.fused_gemv_q6k(input_vec, q_weight_data, D_in, D_out)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


def fused_gemv_q3k(input_vec, q_weight_data, D_in, D_out, dtype=None):
    """Q3_K fused GEMV using dp4a CUDA kernel."""
    if not _CUDA_KERNELS_AVAILABLE:
        raise RuntimeError("CUDA GGUF kernels not available")
    out = _cuda_ext.fused_gemv_q3k(input_vec, q_weight_data, D_in, D_out)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out

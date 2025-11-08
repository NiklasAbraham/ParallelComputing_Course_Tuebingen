"""
Futhark wrapper that allows selecting between CPU (POCL) and GPU (NVIDIA) backends.

Usage:
    from futhark_wrapper import get_futhark_kernel
    
    # CPU backend (POCL)
    kernel_cpu = get_futhark_kernel(backend="cpu")
    
    # GPU backend (NVIDIA OpenCL)
    kernel_gpu = get_futhark_kernel(backend="gpu")
"""

import os
from pathlib import Path
from typing import Literal, Optional


def get_futhark_kernel(backend: Literal["cpu", "gpu", "cuda"] = "cpu"):
    """
    Get a Futhark kernel instance configured for the specified backend.
    
    Args:
        backend: Either "cpu" (POCL), "gpu" (NVIDIA OpenCL), or "cuda" (NVIDIA CUDA)
        
    Returns:
        A Futhark kernel instance with the sum_squares entry point
        
    Raises:
        RuntimeError: If the backend initialization fails
    """
    if backend not in ("cpu", "gpu", "cuda"):
        raise ValueError(f"Invalid backend '{backend}'. Must be 'cpu', 'gpu', or 'cuda'.")
    
    # Set environment variables before importing
    original_env = {}
    
    if backend == "cpu":
        # Force POCL CPU backend
        original_env["OCL_ICD_VENDORS"] = os.environ.get("OCL_ICD_VENDORS")
        original_env["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX")

        pocl_env = original_env["OCL_ICD_VENDORS"]
        if not pocl_env or "pocl" not in pocl_env:
            candidates = [
                Path(pocl_env) if pocl_env else None,
                Path("/etc/OpenCL/vendors"),
                Path("/usr/lib/x86_64-linux-gnu/opencl-vendors"),
            ]
            # Search nix store for pocl once using glob
            nix_candidates = Path("/nix/store").glob("*pocl*/etc/OpenCL/vendors")
            candidates.extend(nix_candidates)

            for candidate in candidates:
                if not candidate:
                    continue
                if candidate.is_file() and candidate.suffix == ".icd":
                    os.environ["OCL_ICD_VENDORS"] = str(candidate.parent)
                    break
                if candidate.is_dir():
                    # Check for pocl icd file inside
                    for icd in candidate.glob("pocl*.icd"):
                        os.environ["OCL_ICD_VENDORS"] = str(candidate)
                        break
                    if os.environ.get("OCL_ICD_VENDORS"):
                        break
        else:
            os.environ["OCL_ICD_VENDORS"] = pocl_env

        os.environ["PYOPENCL_CTX"] = "portable"
        
    elif backend == "gpu":
        # Use NVIDIA OpenCL
        original_env["OCL_ICD_VENDORS"] = os.environ.get("OCL_ICD_VENDORS")
        original_env["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX")
        
        # Point to system NVIDIA ICD
        if os.path.exists("/etc/OpenCL/vendors"):
            os.environ["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"
        os.environ["PYOPENCL_CTX"] = "0"  # Select first device (usually NVIDIA)
    
    elif backend == "cuda":
        # Use NVIDIA CUDA backend
        try:
            from sample_kernel_cuda import SampleKernelCUDA
            return SampleKernelCUDA()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CUDA backend: {e}\n"
                "Make sure libsample_kernel.so is built and CUDA drivers are installed."
            ) from e
    
    try:
        # Import the sample_kernel module (PyOpenCL)
        import sample_kernel
        
        # Create kernel instance
        if backend == "gpu":
            # Try to initialize with NVIDIA device
            try:
                kernel = sample_kernel.sample_kernel(device_pref="NVIDIA")
                return kernel
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize GPU backend: {e}\n"
                    "The NVIDIA OpenCL compiler may not be working properly."
                ) from e
        else:
            # CPU backend
            kernel = sample_kernel.sample_kernel()
            return kernel
            
    except Exception as e:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        raise
    finally:
        # Note: We don't restore environment here because the module is already loaded
        pass


def benchmark_backend(backend: Literal["cpu", "gpu", "cuda"], data, num_runs: int = 100):
    """
    Benchmark a specific Futhark backend.
    
    Args:
        backend: Either "cpu", "gpu", or "cuda"
        data: NumPy array to process
        num_runs: Number of iterations
        
    Returns:
        Tuple of (average_time, result)
    """
    import time
    import numpy as np
    
    try:
        kernel = get_futhark_kernel(backend=backend)
    except RuntimeError as e:
        print(f"Warning: {e}")
        return None, None
    
    # Warm-up
    try:
        warmup_result = kernel.sum_squares(data)
    except Exception as e:
        print(f"Warning: {backend} backend failed during warm-up: {e}")
        return None, None
    
    # Benchmark
    times = []
    results = []
    for _ in range(num_runs):
        start_time = time.time()
        result = kernel.sum_squares(data)
        end_time = time.time()
        times.append(end_time - start_time)
        results.append(result)
    
    avg_time = sum(times) / len(times)
    final_result = results[-1]
    
    return avg_time, final_result


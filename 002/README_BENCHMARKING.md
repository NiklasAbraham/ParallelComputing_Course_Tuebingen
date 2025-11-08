# Futhark + PyTorch Benchmarking Guide

## Quick Start

### Option 1: Simple Benchmark (CPU Futhark vs GPU PyTorch)

```bash
nix develop
python kernel_test.py
```

This compares:
- **Futhark**: PyOpenCL with POCL (CPU, multi-core)
- **PyTorch**: CUDA (GPU)

### Option 2: Multi-Backend Benchmark (with GPU attempt)

```bash
nix develop
python kernel_test_multi.py
```

This attempts to benchmark:
- **Futhark CPU**: PyOpenCL with POCL
- **Futhark GPU**: PyOpenCL with NVIDIA (currently fails due to compiler issue)
- **PyTorch GPU**: CUDA

## Architecture

### Futhark Wrapper (`futhark_wrapper.py`)

Provides a unified interface to select Futhark backends:

```python
from futhark_wrapper import get_futhark_kernel

# CPU backend (POCL)
kernel_cpu = get_futhark_kernel(backend="cpu")
result = kernel_cpu.sum_squares(data)

# GPU backend (NVIDIA OpenCL) - currently not working
try:
    kernel_gpu = get_futhark_kernel(backend="gpu")
    result = kernel_gpu.sum_squares(data)
except RuntimeError as e:
    print(f"GPU backend unavailable: {e}")
```

The wrapper automatically:
- Sets appropriate environment variables (`OCL_ICD_VENDORS`, `PYOPENCL_CTX`)
- Initializes the correct OpenCL platform
- Handles backend-specific errors gracefully

### Test Scripts

1. **`kernel_test.py`**: Simple CPU vs GPU comparison (always works)
2. **`kernel_test_multi.py`**: Attempts all backends, reports which ones succeed

## Current Status

### Working ✓
- **Futhark CPU (POCL)**: ~0.134 seconds for 100M elements
- **PyTorch CUDA**: ~0.003 seconds for 100M elements
- **Speedup**: PyTorch is ~45x faster than Futhark CPU

### Not Working ✗
- **Futhark GPU (NVIDIA OpenCL)**: Compiler fails with empty error log

The NVIDIA OpenCL compiler (part of driver 550.163.01) fails to compile even
simple kernels. The error is:

```
clBuildProgram failed: BUILD_PROGRAM_FAILURE
Build log: (empty)
```

This affects:
- Futhark-generated OpenCL kernels
- Even trivial hand-written OpenCL kernels

### Why PyTorch CUDA Works But Futhark OpenCL Doesn't

- **PyTorch**: Uses CUDA runtime API directly (bypasses OpenCL)
- **Futhark OpenCL**: Uses OpenCL C compiler in the NVIDIA driver
- **Issue**: The OpenCL compiler component in driver 550.163.01 appears broken

## Potential Solutions

### 1. Use Futhark CUDA Backend (attempted, failed)

**Problem**: Version mismatch between host driver (CUDA 12.4) and Nix toolkit (CUDA 12.8)

```bash
futhark cuda --library sample_kernel.fut
# Fails: PTX compiled with unsupported toolchain (error 222)
```

**Would need**: Either upgrade driver to 12.8 or install CUDA 12.4 toolkit on host

### 2. Fix NVIDIA OpenCL Compiler (unknown how)

The OpenCL compiler in the driver is broken. Possible causes:
- Missing dependencies
- Incompatible compiler toolchain
- Driver bug

**Would need**: Investigation or driver update

### 3. Use AMD GPU with ROCm OpenCL (if available)

AMD's ROCm OpenCL implementation is generally more reliable than NVIDIA's.

### 4. Accept CPU-only Futhark for now

Current working configuration provides a valid baseline for:
- Comparing Futhark programming model vs PyTorch
- Understanding performance characteristics
- Testing algorithmic correctness

## Modifying Kernels

Edit `sample_kernel.fut`:

```futhark
entry sum_squares [n] (xs: [n]f64) : f64 =
  reduce (+) 0.0 (map (\x -> x*x) xs)
```

Recompile:

```bash
futhark pyopencl --library sample_kernel.fut
mv sample_kernel sample_kernel.py
```

Run benchmark:

```bash
python kernel_test.py  # or kernel_test_multi.py
```

## Environment Details

The `nix develop` shell provides:
- Futhark 0.25.33
- Python 3.12 with NumPy, PyOpenCL, PyTorch (CUDA-enabled)
- POCL (Portable OpenCL) for CPU execution
- CUDA toolkit 12.8 (headers only, runtime from host)
- Host NVIDIA driver 550.163.01 (CUDA 12.4)

Environment variables set by shell hook:
- `OCL_ICD_VENDORS`: Points to POCL by default
- `PYOPENCL_CTX`: Set to "portable" for CPU
- `CUDA_PATH`: Points to Nix CUDA toolkit
- `LD_LIBRARY_PATH`: Includes host NVIDIA libraries for PyTorch

## Performance Notes

For the `sum_squares` kernel on 100M float64 elements:

| Backend | Time | Relative |
|---------|------|----------|
| Futhark CPU (POCL) | ~0.134s | 1.0x |
| Futhark CUDA | ~0.123s | 1.1x faster |
| PyTorch CUDA | ~0.003s | 45x faster |

The large gap is expected:
- GPU has massive parallel compute (10,496 CUDA cores on RTX 3090)
- Memory bandwidth: ~936 GB/s (GPU) vs ~50 GB/s (DDR4)
- This is a memory-bound operation ideal for GPU

If Futhark OpenCL worked, we'd expect it to be competitive with PyTorch CUDA
(both would be memory-bandwidth limited).

For a compute-heavy kernel (matrix multiply + bias + ReLU + sum, 1024×1024 matrices):

| Backend | Time | Speedup vs Futhark CPU |
|---------|------|------------------------|
| Futhark CPU (POCL) | ~0.643s | 1.0x |
| Futhark CUDA | ~0.007s | 89.9x |
| PyTorch CUDA | ~0.005s | 127.4x |

Run with:

```bash
python kernel_test_matmul.py
```


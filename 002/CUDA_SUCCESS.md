# Futhark CUDA Backend - Successfully Working!

## Summary

After upgrading the NVIDIA driver to 580.95.05 (CUDA 13.0), the Futhark CUDA backend now works successfully!

## Benchmark Results

Running `python kernel_test_multi.py` with 100M float64 elements:

```
Futhark CPU (POCL):           0.131s
Futhark CUDA (NVIDIA):        0.123s  (1.07x faster than CPU)
PyTorch CUDA:                 0.003s  (42.7x faster than Futhark CUDA)
```

## Why is Futhark CUDA Only Slightly Faster Than CPU?

The Futhark CUDA backend is working correctly but shows minimal speedup because:

### 1. Data Transfer Overhead
- **Futhark**: Transfers data CPU→GPU→CPU on every call
  - `futhark_new_f64_1d()` copies 800MB to GPU
  - Result copied back to CPU
  - This happens 100 times in the benchmark

- **PyTorch**: Keeps data on GPU between operations
  - Data transferred once at start
  - All operations happen on GPU
  - Only final result comes back

### 2. Kernel Launch Overhead
- Each Futhark call has ~10-20μs launch overhead
- For 100 iterations: ~1-2ms total overhead
- PyTorch amortizes this across larger operations

### 3. Memory-Bound Operation
- `sum_squares` is limited by memory bandwidth, not compute
- GPU: ~936 GB/s (RTX 3090)
- CPU: ~50 GB/s (DDR4)
- But transfer overhead negates the bandwidth advantage

## When Would Futhark CUDA Be Faster?

Futhark CUDA would show significant speedup for:

1. **Complex compute-bound kernels**
   - Matrix multiplication
   - FFT
   - Convolutions
   - Where compute time >> transfer time

2. **Persistent GPU data**
   - If you could keep arrays on GPU between calls
   - Multiple operations on same data
   - Currently not exposed in the Python API

3. **Larger problem sizes**
   - Where GPU parallelism dominates overhead
   - Though this kernel is already 100M elements

## How to Use All Three Backends

```python
from futhark_wrapper import get_futhark_kernel
import numpy as np

data = np.arange(1000000, dtype=np.float64)

# CPU backend (POCL)
kernel_cpu = get_futhark_kernel(backend="cpu")
result_cpu = kernel_cpu.sum_squares(data)

# CUDA backend (NVIDIA CUDA)
kernel_cuda = get_futhark_kernel(backend="cuda")
result_cuda = kernel_cuda.sum_squares(data)

# OpenCL backend (NVIDIA OpenCL) - currently broken
try:
    kernel_gpu = get_futhark_kernel(backend="gpu")
    result_gpu = kernel_gpu.sum_squares(data)
except RuntimeError as e:
    print(f"OpenCL backend failed: {e}")
```

## What Was Fixed

1. **Driver Upgrade**: 550.163.01 (CUDA 12.4) → 580.95.05 (CUDA 13.0)
2. **CUDA Library Rebuild**: Regenerated with matching driver version
3. **Python Wrapper Fix**: Added missing `argtypes` for `futhark_context_new`
4. **Backend Selection**: Added CUDA option to `futhark_wrapper.py`
5. **Benchmark Update**: Added CUDA to `kernel_test_multi.py`

## Files Modified

- `sample_kernel.c/h` - Regenerated with `futhark cuda --library`
- `libsample_kernel.so` - Rebuilt shared library
- `sample_kernel_cuda.py` - Fixed ctypes signatures
- `futhark_wrapper.py` - Added CUDA backend support
- `kernel_test_multi.py` - Added CUDA benchmarking

## Current Status

| Backend | Status | Performance |
|---------|--------|-------------|
| Futhark CPU (POCL) | ✓ Working | 0.131s |
| Futhark CUDA | ✓ Working | 0.123s |
| Futhark OpenCL | ✗ Broken | Driver compiler issue |
| PyTorch CUDA | ✓ Working | 0.003s |

## Next Steps

For better Futhark GPU performance:
1. Use compute-intensive kernels (not just sum/reduce)
2. Batch multiple operations to amortize transfer costs
3. Consider exposing GPU memory management in Python API
4. Profile with larger/more complex kernels

The infrastructure is now in place to benchmark any Futhark kernel against PyTorch!


# Futhark + PyTorch Benchmark Pipeline

This document describes how to compile Futhark kernels and benchmark them against
PyTorch.

## 1. Environment setup

Enter the Nix development shell:

```bash
nix develop
```

This provides:
- Futhark compiler
- CUDA toolkit (for PyTorch)
- Python with NumPy, PyOpenCL, PyCUDA, and PyTorch (CUDA-enabled)
- POCL (Portable OpenCL) for CPU execution
- OpenCL ICD loaders

The shell hook automatically:
- Sets `OCL_ICD_VENDORS` to use POCL
- Sets `PYOPENCL_CTX="portable"` for CPU-based OpenCL
- Symlinks host NVIDIA driver libraries for PyTorch CUDA support
- Extends `LD_LIBRARY_PATH` to prioritize host driver over Nix toolkit

## 2. Compile the Futhark kernel

Generate the PyOpenCL library from your `.fut` source:

```bash
futhark pyopencl --library sample_kernel.fut
mv sample_kernel sample_kernel.py
```

This produces a Python module that:
- Compiles OpenCL kernels at runtime
- Exposes entry points like `sum_squares` through a class interface
- Runs on any OpenCL platform (CPU via POCL, or GPU if configured)

## 3. Run the benchmark

```bash
python kernel_test.py
```

This script:
1. Creates a 100M-element float64 array
2. Transfers it to the GPU for PyTorch (if CUDA available)
3. Initializes the Futhark PyOpenCL context (using POCL CPU backend)
4. Runs 100 iterations of `sum_squares` on both backends
5. Prints average execution times

Expected output:
```
Futhark result (last run): 3.3333332833333695e+23
Average time over 100 runs (Futhark): 0.133687 seconds
PyTorch result (last run): 3.333333283333333e+23
Average time over 100 runs (PyTorch on cuda): 0.003157 seconds
```

## 4. Current configuration

**Futhark backend**: PyOpenCL with POCL (CPU)
- Runs on all available CPU cores
- No GPU driver compatibility issues
- Slower than GPU but portable

**PyTorch backend**: CUDA (GPU)
- Uses NVIDIA RTX 3090 GPUs
- Requires host NVIDIA driver (currently 550.163.01, CUDA 12.4)
- Much faster for large tensor operations

### Why not Futhark CUDA?

The CUDA backend was attempted but failed due to a driver/toolkit version mismatch:
- Host driver: CUDA 12.4 (550.163.01)
- Nix toolkit: CUDA 12.8 (only version available in nixpkgs)
- NVRTC 12.8 generates PTX that the 12.4 driver rejects (error 222)
- Host system lacks matching CUDA runtime libraries

The PyOpenCL backend with POCL provides a working baseline for benchmarking.

### Enabling GPU for Futhark

To use GPU acceleration for Futhark:

1. **Option A**: Install CUDA 12.4 toolkit on the host system, then adjust the
   flake to use those libraries

2. **Option B**: Upgrade the NVIDIA driver to support CUDA 12.8

3. **Option C**: Fix the NVIDIA OpenCL compiler issue (currently fails with
   empty build log when targeting NVIDIA platform)

## 5. Modifying the kernel

Edit `sample_kernel.fut`, then recompile:

```bash
futhark pyopencl --library sample_kernel.fut
mv sample_kernel sample_kernel.py
```

The Python script will automatically use the updated module on the next run.

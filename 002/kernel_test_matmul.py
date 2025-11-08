"""
Benchmark a heavier kernel (matrix multiplication + bias + ReLU + sum)
across Futhark CPU, Futhark CUDA, and PyTorch CUDA backends.
"""

import time
import numpy as np
import torch

import sample_kernel  # PyOpenCL backend
from sample_kernel_cuda import SampleKernelCUDA  # CUDA backend

# Problem dimensions (adjust if memory-limited)
N = 1024 * 10
K = 1024 * 10
M = 1024 * 10
NUM_RUNS = 20

# Generate deterministic data for reproducibility
rng = np.random.default_rng(seed=42)
A = rng.standard_normal((N, K), dtype=np.float64)
B = rng.standard_normal((K, M), dtype=np.float64)
BIAS = rng.standard_normal((M,), dtype=np.float64)

# PyTorch setup
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_t = torch.from_numpy(A).to(torch_device).to(torch.float64)
B_t = torch.from_numpy(B).to(torch_device).to(torch.float64)
BIAS_t = torch.from_numpy(BIAS).to(torch_device).to(torch.float64)


def torch_kernel() -> float:
    out = torch.matmul(A_t, B_t)
    out = torch.add(out, BIAS_t)
    out = torch.relu(out)
    val = out.sum()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    return val.item()


def run_benchmark(name: str, func, warmup: int = 1):
    for _ in range(warmup):
        func()

    times = []
    result = None
    for _ in range(NUM_RUNS):
        start = time.time()
        result = func()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time, result


print("Preparing Futhark CPU backend (PyOpenCL)...")
# futhark_cpu = sample_kernel.sample_kernel()
# futhark_cpu_func = lambda: futhark_cpu.matmul_bias_relu_sum(A, B, BIAS)

print("Preparing Futhark CUDA backend...")
futhark_cuda = SampleKernelCUDA()
futhark_cuda_func = lambda: futhark_cuda.matmul_bias_relu_sum(A, B, BIAS)

print("Preparing PyTorch backend...")
torch_func = torch_kernel

print("\nRunning benchmarks...\n")
# cpu_time, cpu_result = run_benchmark("Futhark CPU", futhark_cpu_func)
cuda_time, cuda_result = run_benchmark("Futhark CUDA", futhark_cuda_func)
torch_time, torch_result = run_benchmark("PyTorch CUDA", torch_func)

print("=" * 70)
print("BENCHMARK RESULTS (Matrix Multiply + Bias + ReLU + Sum)")
print("=" * 70)
# print(f"Futhark CPU (POCL):       {cpu_time:.6f} s")
print(f"Futhark CUDA (CUDA):      {cuda_time:.6f} s")
print(f"PyTorch CUDA:             {torch_time:.6f} s")

print("\nResults comparison:")
# print(f"  Futhark CPU result:  {cpu_result}")
print(f"  Futhark CUDA result: {cuda_result}")
print(f"  PyTorch result:      {torch_result}")

print("\nSpeedups:")
# print(f"  CUDA vs CPU:         {cpu_time / cuda_time:.2f}x faster")
# print(f"  PyTorch vs CPU:      {cpu_time / torch_time:.2f}x faster")
print(f"  PyTorch vs Futhark CUDA: {cuda_time / torch_time:.2f}x faster")

futhark_cuda.close()


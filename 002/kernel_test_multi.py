"""
Benchmark script comparing Futhark (CPU and GPU) against PyTorch (GPU).
"""

import time
import numpy as np
import torch
from futhark_wrapper import benchmark_backend

# 1. Create test data
data = np.arange(100_000_000, dtype=np.float64)

# 2. PyTorch benchmark (GPU)
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_data = torch.from_numpy(data).to(torch_device)
torch_data = torch_data.to(torch.float64)

def torch_sum_squares() -> float:
    tmp = torch.sum(torch_data * torch_data)
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    return tmp.item()

num_runs = 100

# Warm-up
_ = torch_sum_squares()

torch_times = []
torch_results = []
for _ in range(num_runs):
    start_time_torch = time.time()
    torch_result = torch_sum_squares()
    end_time_torch = time.time()
    torch_times.append(end_time_torch - start_time_torch)
    torch_results.append(torch_result)

torch_result = torch_results[-1]
avg_torch_time = sum(torch_times) / len(torch_times)

# 3. Futhark CPU benchmark
print("Benchmarking Futhark CPU backend...")
futhark_cpu_time, futhark_cpu_result = benchmark_backend("cpu", data, num_runs)

# 4. Futhark GPU benchmark (OpenCL)
print("Benchmarking Futhark GPU backend (OpenCL)...")
futhark_gpu_time, futhark_gpu_result = benchmark_backend("gpu", data, num_runs)

# 5. Futhark CUDA benchmark
print("Benchmarking Futhark CUDA backend...")
futhark_cuda_time, futhark_cuda_result = benchmark_backend("cuda", data, num_runs)

# 6. Print results
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

if futhark_cpu_result is not None:
    print(f"\nFuthark CPU (POCL):")
    print(f"  Result: {futhark_cpu_result}")
    print(f"  Average time: {futhark_cpu_time:.6f} seconds")
else:
    print(f"\nFuthark CPU (POCL): FAILED")

if futhark_gpu_result is not None:
    print(f"\nFuthark GPU (NVIDIA OpenCL):")
    print(f"  Result: {futhark_gpu_result}")
    print(f"  Average time: {futhark_gpu_time:.6f} seconds")
else:
    print(f"\nFuthark GPU (NVIDIA OpenCL): FAILED")

if futhark_cuda_result is not None:
    print(f"\nFuthark CUDA (NVIDIA CUDA):")
    print(f"  Result: {futhark_cuda_result}")
    print(f"  Average time: {futhark_cuda_time:.6f} seconds")
else:
    print(f"\nFuthark CUDA (NVIDIA CUDA): FAILED")

print(f"\nPyTorch ({torch_device.type.upper()}):")
print(f"  Result: {torch_result}")
print(f"  Average time: {avg_torch_time:.6f} seconds")

# 7. Speedup comparison
print("\n" + "="*70)
print("SPEEDUP ANALYSIS")
print("="*70)

if futhark_cpu_time and futhark_cuda_time:
    speedup_cuda_vs_cpu = futhark_cpu_time / futhark_cuda_time
    print(f"\nFuthark CUDA vs CPU: {speedup_cuda_vs_cpu:.2f}x faster")

if futhark_cpu_time and futhark_gpu_time:
    speedup_gpu_vs_cpu = futhark_cpu_time / futhark_gpu_time
    print(f"Futhark OpenCL vs CPU: {speedup_gpu_vs_cpu:.2f}x faster")

if futhark_cuda_time and torch_device.type == "cuda":
    speedup_torch_vs_futhark = futhark_cuda_time / avg_torch_time
    print(f"PyTorch CUDA vs Futhark CUDA: {speedup_torch_vs_futhark:.2f}x faster")

if futhark_cpu_time and torch_device.type == "cuda":
    speedup_torch_vs_cpu = futhark_cpu_time / avg_torch_time
    print(f"PyTorch CUDA vs Futhark CPU: {speedup_torch_vs_cpu:.2f}x faster")

print("\n" + "="*70)


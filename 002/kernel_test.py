import time

import numpy as np
import sample_kernel  # Import the generated Python file
import torch

# 1. Create your data in Python using NumPy
data = np.arange(100_000_000, dtype=np.float64)

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_data = torch.from_numpy(data).to(torch_device)
torch_data = torch_data.to(torch.float64)

# 2. Initialize the Futhark context
# This finds the OpenCL device, compiles the C code, etc.
futhark_kernel = sample_kernel.sample_kernel()

num_runs = 100

# Warm-up run to avoid setup costs in timing loop
warmup_result = futhark_kernel.sum_squares(data)

futhark_times = []
futhark_results = []
for _ in range(num_runs):
    start_time = time.time()
    result = futhark_kernel.sum_squares(data)
    end_time = time.time()
    futhark_times.append(end_time - start_time)
    futhark_results.append(result)

result = futhark_results[-1]
avg_futhark_time = sum(futhark_times) / len(futhark_times)


def torch_sum_squares() -> float:
    tmp = torch.sum(torch_data * torch_data)
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    return tmp.item()


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

print(f"Futhark result (last run): {result}")
print(f"Average time over {num_runs} runs (Futhark): {avg_futhark_time:.6f} seconds")
print(f"PyTorch result (last run): {torch_result}")
print(
    f"Average time over {num_runs} runs (PyTorch on {torch_device.type}): {avg_torch_time:.6f} seconds"
)

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
  """Initializes the distributed process group."""
  # 1. Set Master Address/Port so workers can find each other
  master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
  master_port = os.environ.get("MASTER_PORT", "29500")

  os.environ["MASTER_ADDR"] = master_addr
  os.environ["MASTER_PORT"] = master_port

  # 2. Initialize the process group
  dist.init_process_group(backend, rank=rank, world_size=world_size)

  # 3. For NCCL, ensure the process uses the correct unique GPU
  if backend == "nccl":
    torch.cuda.set_device(rank)


def run_benchmark(rank, world_size, backend, data_size_bytes):
  """Main benchmarking function run by each process."""
  setup(rank, world_size, backend)
  num_elements = data_size_bytes // 4
  device = torch.device("cuda" if backend == "nccl" else "cpu")
  data = torch.rand((num_elements,), dtype=torch.float32, device=device)

  # --- 1. Warmup ---
  for _ in range(5):
    dist.all_reduce(data, async_op=False)

  if backend == "nccl":
    torch.cuda.synchronize()

  # --- 2. Measurement ---
  start_time = time.perf_counter()

  for _ in range(10):
    dist.all_reduce(data, async_op=False)

  # Synchronize AFTER the loop to ensure GPU finished the work
  if backend == "nccl":
    torch.cuda.synchronize()

  end_time = time.perf_counter()

  # Calculate average time
  avg_time = (end_time - start_time) / 10
  # Only Rank 0 prints to avoid duplicate output

  # Determine the device label based on the backend
  # Since NCCL is for GPUs and Gloo is typically used for CPUs in this context
  device_label = "GPUs" if backend == "nccl" else "CPUs"

  # Format data size for readability (e.g., 1MB, 10MB, 100MB, 1GB)
  if data_size_bytes >= 1024**3:
      size_str = f"{data_size_bytes / (1024**3):.0f}GB"
  elif data_size_bytes >= 1024**2:
      size_str = f"{data_size_bytes / (1024**2):.0f}MB"
  else:
      size_str = f"{data_size_bytes} bytes"

  # Only Rank 0 prints to avoid duplicate output
  if rank == 0:
      print(
          f"Config: {backend} | {world_size} {device_label} | {size_str} |"
          f" Time: {avg_time:.6f}s"
      )

  # --- 4. Cleanup ---
  dist.destroy_process_group()


if __name__ == "__main__":
  # Define the sweep configurations
  backends = ["gloo", "nccl"]
  process_counts = [2, 4, 6]
  data_sizes = [
      1024**2,
      10 * 1024**2,
      100 * 1024**2,
      1024**3,
  ]  # 1MB, 10MB, 100MB, 1GB

  for backend in backends:
    for world_size in process_counts:
      # Skip invalid configurations (e.g., NCCL with insufficient GPUs)
      if backend == "nccl" and torch.cuda.device_count() < world_size:
        print(
            f"Skipping {world_size} GPUs (only {torch.cuda.device_count()}"
            " available)"
        )
        continue

      for size in data_sizes:
        # Spawn the processes
        mp.spawn(
            run_benchmark,
            args=(world_size, backend, size),
            nprocs=world_size,
            join=True,
        )

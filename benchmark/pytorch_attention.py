import argparse
import os
import timeit
# Local import
# Ensure this path is in your PYTHONPATH or run from the correct directory
from cs336_basics.cs336_basics.model import scaled_dot_product_attention
import einx
import pandas as pd
import torch

# --- Constants ---

BATCH_SIZE = 8
# D_HEAD is determined dynamically: d_model // num_heads
NUM_HEADS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Benchmark Configurations
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


# --- Classes & Functions ---


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(description='Benchmark Attention Mechanism')
  # To run with torch.compile enabled, use the flag: --compile
  # Example: python script_name.py --compile
  parser.add_argument(
      '--compile',
      action='store_true',
      help='Enable torch.compile for the attention function',
  )
  return parser.parse_args()


class AttentionBenchmarkRunner:
  """A class to benchmark the performance of the attention mechanism."""

  def __init__(
      self,
      d_model: int,
      seq_len: int,
      batch_size: int,
      num_heads: int,
      device: torch.device,
      jit_compile: bool = False,
  ):
    """Initializes the benchmark runner.

    Args:
        d_model: The embedding dimension of each attention head.
        seq_len: The sequence length of the input.
        batch_size: The batch size.
        num_heads: The number of attention heads.
        device: The device to run the benchmark on.
        jit_compile: Whether to use torch.compile.
    """
    self.d_model = d_model
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_heads = num_heads
    self.device = device
    self.d_head = self.d_model // self.num_heads

    # Select and optionally compile the attention function
    self.attn_fn = scaled_dot_product_attention
    if jit_compile:
      print(f'Compiling attention function for d_model={d_model}...')
      self.attn_fn = torch.compile(self.attn_fn)

    # Initialize tensors
    self.q = torch.randn(
        self.batch_size,
        self.seq_len,
        self.num_heads,
        self.d_head,
        device=self.device,
        requires_grad=True,
    )
    self.k = torch.randn(
        self.batch_size,
        self.seq_len,
        self.num_heads,
        self.d_head,
        device=self.device,
        requires_grad=True,
    )
    self.v = torch.randn(
        self.batch_size,
        self.seq_len,
        self.num_heads,
        self.d_head,
        device=self.device,
        requires_grad=True,
    )

  def forward(self):
    """Runs the forward pass."""
    return self.attn_fn(self.q, self.k, self.v)

  def backward(self, output):
    """Runs the backward pass using the provided output."""
    output.sum().backward()


def benchmark(
    d_model: int,
    seq_len: int,
    jit_compile: bool,
    batch_size: int = BATCH_SIZE,
    num_heads: int = NUM_HEADS,
    warmup: int = 10,
    steps: int = 100,
):
  """Runs the benchmark for a given configuration."""

  # Initialize runner
  runner = AttentionBenchmarkRunner(
      d_model,
      seq_len,
      batch_size=batch_size,
      num_heads=num_heads,
      device=DEVICE,
      jit_compile=jit_compile,
  )

  # Warmup Phase
  for _ in range(warmup):
    output = runner.forward()
    torch.cuda.synchronize()
    runner.backward(output)
    torch.cuda.synchronize()

  # Measurement Phase
  forward_time = 0
  backward_time = 0
  pre_back_memory = 0

  for _ in range(steps):
    torch.cuda.synchronize()

    # Measure Forward
    forward_start_time = timeit.default_timer()
    output = runner.forward()
    torch.cuda.synchronize()
    forward_time += timeit.default_timer() - forward_start_time

    # Measure Memory (Activations)
    pre_back_memory += torch.cuda.memory_allocated() / 1024**2

    # Measure Backward
    backward_start_time = timeit.default_timer()
    runner.backward(output)
    torch.cuda.synchronize()
    backward_time += timeit.default_timer() - backward_start_time

  return (
      round(forward_time / steps * 1000, 2),  # ms
      round(backward_time / steps * 1000, 2),  # ms
      round(pre_back_memory / steps, 2),  # MB
  )


# --- Main Execution ---


def main():
  args = parse_args()

  # Initialize results DataFrame
  results = pd.DataFrame(
      columns=[
          'd_model',
          'seq_len',
          'forward_time_ms',
          'backward_time_ms',
          'pre_back_memory_mb',
      ]
  )

  print(f'Starting benchmark (JIT Compile: {args.compile})...')

  for d_model in D_MODELS:
    for seq_len in SEQ_LENS:
      try:
        forward_time, backward_time, pre_back_memory = benchmark(
            d_model=d_model, seq_len=seq_len, jit_compile=args.compile
        )

        results.loc[len(results)] = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_ms': forward_time,
            'backward_time_ms': backward_time,
            'pre_back_memory_mb': pre_back_memory,
        }
      except Exception as e:
        print(f'Error benchmarking d_model={d_model}, seq_len={seq_len}: {e}')
        results.loc[len(results)] = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_ms': None,
            'backward_time_ms': None,
            'pre_back_memory_mb': None,
        }
        # Reset memory just in case
        torch.cuda.empty_cache()

  # --- Output Results ---

  results = results.reset_index(drop=True)
  markdown_table = results.to_markdown()
  print('\nResults:')
  print(markdown_table)

  # Save to file
  filename = f'pytorch_attn_{"jit" if args.compile else "nojit"}.txt'
  # Ensure directory exists or handle path
  snapshot_dir = '/xcloud-shared/jiayulu'
  if os.path.exists(snapshot_dir):
    snapshot_path = os.path.join(snapshot_dir, filename)
    with open(snapshot_path, 'w') as f:
      f.write(markdown_table)
    print(f'\nResults saved to {snapshot_path}')
  else:
    print(
        f'\nDirectory {snapshot_dir} not found. Results printed to stdout only.'
    )


if __name__ == '__main__':
  # ---------------------------------------------------------
  # HOW TO RUN:
  # Standard run: python benchmark.py
  # With Compile: python benchmark.py --compile
  # ---------------------------------------------------------
  main()

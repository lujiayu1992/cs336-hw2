import os
import timeit
from cs336_basics.cs336_basics.model import scaled_dot_product_attention
import einx
import pandas as pd
import torch

# --- Constants ---

BATCH_SIZE = 8
NUM_HEADS = 1
device = torch.device('cuda')
jit_compile = False
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


class AttentionBenchmarkRunner:
  """A class to benchmark the performance of the attention mechanism."""

  def __init__(
      self,
      d_model: int,
      seq_len: int,
      batch_size: int = 8,
      num_heads: int = 1,
      device: torch.device = torch.device('cuda'),
  ):
    """Initializes the benchmark runner.

    Args:
        d_model: The embedding dimension of each attention head.
        seq_len: The sequence length of the input.
        batch_size: The batch size.
        num_heads: The number of attention heads.
        device: The device to run the benchmark on.
    """
    self.d_model = d_model
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_heads = num_heads
    self.device = device
    self.d_head = self.d_model // self.num_heads
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

  def _forward(self):
    """Runs the forward pass of the attention mechanism."""
    _ = scaled_dot_product_attention(
        self.q,
        self.k,
        self.v,
    )

  def _backward(self):
    """Runs the backward pass of the attention mechanism."""
    attention_scores = scaled_dot_product_attention(
        self.q,
        self.k,
        self.v,
    )
    attention_scores.sum().backward()
    self.q.grad.zero_()
    self.k.grad.zero_()
    self.v.grad.zero_()


# --- Benchmark Function ---


def benchmark(
    d_model: int,
    seq_len: int,
    batch_size: int = 8,
    num_heads: int = 1,
    warmup: int = 10,
    steps: int = 100,
):
  """This function runs the benchmark for a given d_model and sequence length.

  It creates random inputs, a causal mask, and then measures the forward and
  backward pass times and memory usage.
  """
  # (Implementation details omitted for skeleton)
  forward_time = 0
  backward_time = 0
  pre_back_memory = 0
  model = AttentionBenchmarkRunner(
      d_model, seq_len, batch_size=batch_size, num_heads=num_heads
  )
  for _ in range(warmup):
    model._forward()
    torch.cuda.synchronize()
    model._backward()
    torch.cuda.synchronize()
  for _ in range(steps):
    torch.cuda.synchronize()
    forward_start_time = timeit.default_timer()
    model._forward()
    torch.cuda.synchronize()
    forward_time += timeit.default_timer() - forward_start_time
    pre_back_memory += torch.cuda.memory_allocated() / 1024**2  # get it in GB
    torch.cuda.synchronize()
    backward_start_time = timeit.default_timer()
    model._backward()
    torch.cuda.synchronize()
    backward_time += timeit.default_timer() - backward_start_time
  return (
      round(forward_time / steps * 1000),
      round(backward_time / steps * 1000),
      round(pre_back_memory / steps),
  )


# --- Main Execution ---


def main():
  """Main function to run the benchmark across different configurations."""
  results = pd.DataFrame(
      columns=[
          'd_model',
          'seq_len',
          'forward_time_ms',
          'backward_time_ms',
          'pre_back_memory_gb',
      ]
  )

  for d_model in D_MODELS:
    for seq_len in SEQ_LENS:
      try:
        forward_time, backward_time, pre_back_memory = benchmark(
            d_model, seq_len
        )
        results.loc[len(results)] = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_ms': forward_time,
            'backward_time_ms': backward_time,
            'pre_back_memory_gb': pre_back_memory,
        }
      except Exception as e:
        print(f'Error benchmarking d_model={d_model}, seq_len={seq_len}: {e}')
        results.loc[len(results)] = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_ms': None,
            'backward_time_ms': None,
            'pre_back_memory_gb': None,
        }
        continue

  # --- Output Results ---

  # save results as markdown table
  # remove index column
  results = results.reset_index(drop=True)
  print(results.to_markdown())

  # save markdown table to file
  filename = f'pytorch_attn_{"jit" if jit_compile else "nojit"}.txt'
  snapshot_path = os.path.join('/xcloud-shared/jiayulu', filename)
  with open(snapshot_path, 'w') as f:
    f.write(results.to_markdown())


if __name__ == '__main__':
  main()

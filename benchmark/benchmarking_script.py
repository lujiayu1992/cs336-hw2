#!/usr/bin/env python

import argparse
import timeit
import torch
import numpy as np
import torch.nn.functional as F

# Import the model and config from the cs336_basics package
from cs336_basics.cs336_basics.model import BasicsTransformerLM

# uv run python benchmark/benchmarking_script.py --mode forward --model_size small
# ----- BENCHMARKING RESULTS -----
#  Config: Mode=forward, Model=small, ContextLen=256
#  Steps: Warmup=5, Measured=10
#  Average Time: 0.088762 seconds
#  Std Deviation: 0.000485 seconds
# --------------------------------
# uv run python benchmark/benchmarking_script.py --mode forward --model_size small --num_warmup 0
# ----- BENCHMARKING RESULTS -----
#  Config: Mode=forward, Model=small, ContextLen=256
#  Steps: Warmup=0, Measured=10
#  Average Time: 0.137355 seconds
#  Std Deviation: 0.138428 seconds
# --------------------------------
# uv run python benchmark/benchmarking_script.py --mode forward_backward --model_size medium
# ----- BENCHMARKING RESULTS -----
#  Config: Mode=forward_backward, Model=medium, ContextLen=256
#  Steps: Warmup=5, Measured=10
#  Average Time: 0.795688 seconds
#  Std Deviation: 0.003630 seconds
# --------------------------------
# uv run python benchmark/benchmarking_script.py --mode forward_backward --model_size large
# ----- BENCHMARKING RESULTS -----
#  Config: Mode=forward_backward, Model=large, ContextLen=256
#  Steps: Warmup=5, Measured=10
#  Average Time: 1.752168 seconds
#  Std Deviation: 0.023761 seconds
# --------------------------------



def get_model_hparams(size_name: str) -> dict:
    """
    Returns a dictionary of hyperparameters based on Table 1.
    """
    # [cite_start]Specifications from Table 1 [cite: 66]
    configs = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }
    if size_name not in configs:
        raise ValueError(
            f"Unknown model size: {size_name}. Valid sizes are: {list(configs.keys())}"
        )
    return configs[size_name]


def run_forward_step(model, inputs):
    with torch.no_grad():
        _ = model(inputs)


def run_forward_backward_step(model, inputs, vocab_size):
    model.zero_grad()
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, vocab_size), inputs.view(-1))
    loss.backward()


def run_benchmark(args):
    """
    Main function to run the benchmarking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    # --- 1. Initialize model given hyperparameters ---
    # Get the hyperparameter dictionary (d_model, d_ff, etc.)
    model_hparams = get_model_hparams(args.model_size)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=10000.0,  # Default from Assignment 1
        **model_hparams,
    ).to(device)
    # --- 2. Generate a random batch of data ---
    # Per §1.1.2: vocab_size=10,000, batch_size=4
    labels = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    print(f"Data shape (Batch, SeqLen): {labels.shape}")
    # --- 3. Run w warm-up steps ---
    for _ in range(args.num_warmup):
        if args.mode == "forward":
            run_forward_step(model, labels)
        elif args.mode == "forward_backward":
            run_forward_backward_step(model, labels, args.vocab_size)
        torch.cuda.synchronize()

    timings = []
    print(f"Running {args.num_steps} measurement steps...")
    for _ in range(args.num_steps):
        start_time = timeit.default_timer()
        if args.mode == "forward":
            run_forward_step(model, labels)
        elif args.mode == "forward_backward":
            run_forward_backward_step(model, labels, args.vocab_size)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    avg_time = np.mean(timings)
    std_dev = np.std(timings)

    print("\n----- BENCHMARKING RESULTS -----")
    print(
        f" Config: Mode={args.mode}, Model={args.model_size}, ContextLen={args.context_length}"
    )
    print(f" Steps: Warmup={args.num_warmup}, Measured={args.num_steps}")
    print(f" Average Time: {avg_time:.6f} seconds")
    print(f" Std Deviation: {std_dev:.6f} seconds")
    print("--------------------------------\n")


if __name__ == "__main__":
    # [cite_start]Script must support command-line arguments for variations [cite: 69]
    parser = argparse.ArgumentParser(
        description="End-to-End Transformer Benchmarking Script"
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "2.7B"],
        help="Model size specification from Table 1",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="forward_backward",
        choices=["forward", "forward_backward"],
        help="Benchmarking mode: only forward pass, or forward + backward pass [cite: 83]",
    )

    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup steps to run before timing [cite: 83, 88]",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of measurement steps to time [cite: 83, 88]",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Sequence length (context length) of the input data.",
    )

    # Constants defined in section 1.1.2
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4, as per section 1.1.2) [cite: 61]",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size (default: 10000, as per section 1.1.2) [cite: 61]",
    )

    args = parser.parse_args()

    run_benchmark(args)

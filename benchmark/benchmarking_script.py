#!/usr/bin/env python

# PYTHONPATH=. python benchmark/benchmarking_script.py --mode forward --model_size xl --num_steps 1 --record_memory
import argparse
import contextlib
import os
import timeit
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# Local Imports
try:
    from cs336_basics.cs336_basics.model import BasicsTransformerLM
except ImportError:
    print("Warning: Could not import BasicsTransformerLM. Ensure PYTHONPATH is set.")

# --- Configuration Constants ---

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

# --- Benchmarking Class ---

class BenchmarkRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._get_model_config(args.model_size)
        self.model = self._initialize_model()
        
        print(f"Initialized BenchmarkRunner on device: {self.device}")
        print(f"Model Configuration: {args.model_size} {self.config}")

    def _get_model_config(self, size_name: str) -> Dict[str, int]:
        if size_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model size: {size_name}. Valid sizes are: {list(MODEL_CONFIGS.keys())}"
            )
        return MODEL_CONFIGS[size_name]

    def _initialize_model(self) -> torch.nn.Module:
        model = BasicsTransformerLM(
            vocab_size=self.args.vocab_size,
            context_length=self.args.context_length,
            rope_theta=10000.0,
            **self.config,
        ).to(self.device)
        return model

    def _generate_batch(self) -> torch.Tensor:
        return torch.randint(
            0,
            self.args.vocab_size,
            (self.args.batch_size, self.args.context_length),
            device=self.device,
            dtype=torch.long,
        )

    def run_step(self, inputs: torch.Tensor):
        """Executes a single step based on the configured mode."""
        if self.args.mode == "forward":
            with torch.no_grad():
                _ = self.model(inputs)
        
        elif self.args.mode == "forward_backward":
            self.model.zero_grad()
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, self.args.vocab_size), 
                inputs.view(-1)
            )
            loss.backward()

    def _handle_memory_recording(self, step_idx: int):
        """Handles starting and stopping PyTorch memory recording."""
        if not (self.args.record_memory and self.device.type == 'cuda'):
            return

        # Start recording
        torch.cuda.memory._record_memory_history(max_entries=100000)

        # We only save the snapshot after the step is done, usually inside the loop logic
        # But for this specific implementation, we return here and handle saving after execution
        return

    def _save_memory_snapshot(self):
        """Saves the memory snapshot to disk."""
        if not os.path.exists(self.args.snapshot_dir):
            os.makedirs(self.args.snapshot_dir)

        filename = f"memory_snapshot_{self.args.model_size}_{self.args.mode}_{self.args.context_length}.pickle"
        snapshot_path = os.path.join(self.args.snapshot_dir, filename)

        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"Memory snapshot saved to: {snapshot_path}")
            print(f"View at https://pytorch.org/memory_viz")
        except Exception as e:
            print(f"Failed to save memory snapshot: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    def run(self):
        """Main execution loop."""
        inputs = self._generate_batch()
        print(f"Data shape (Batch, SeqLen): {inputs.shape}")

        # Setup Mixed Precision Context
        ctx = contextlib.nullcontext()
        if self.args.use_bf16:
            ctx = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)

        with ctx:
            # 1. Warmup
            print(f"Running {self.args.num_warmup} warmup steps...")
            for _ in range(self.args.num_warmup):
                self.run_step(inputs)
                torch.cuda.synchronize()

            # 2. Measurement
            timings = []
            print(f"Running {self.args.num_steps} measurement steps...")
            
            for i in range(self.args.num_steps):
                # Optional: Start Memory Recording
                if self.args.record_memory and self.device.type == 'cuda':
                     torch.cuda.memory._record_memory_history(max_entries=100000)

                start_time = timeit.default_timer()
                self.run_step(inputs)
                
                # Optional: Save Snapshot (only on last step or specific logic)
                if self.args.record_memory and self.device.type == 'cuda':
                    self._save_memory_snapshot()

                torch.cuda.synchronize()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)

        self._print_results(timings)

    def _print_results(self, timings):
        avg_time = np.mean(timings)
        std_dev = np.std(timings)

        print("\n----- BENCHMARKING RESULTS -----")
        print(f" Config: Mode={self.args.mode}, Model={self.args.model_size}, ContextLen={self.args.context_length}")
        print(f" Steps: Warmup={self.args.num_warmup}, Measured={self.args.num_steps}")
        print(f" Average Time: {avg_time:.6f} seconds")
        print(f" Std Deviation: {std_dev:.6f} seconds")
        print("--------------------------------\n")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="End-to-End Transformer Benchmarking Script")

    # Model Arguments
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=list(MODEL_CONFIGS.keys()), help="Model size specification.")
    parser.add_argument("--mode", type=str, default="forward_backward",
                        choices=["forward", "forward_backward"], help="Benchmarking mode.")
    
    # Timing Arguments
    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup steps.")
    parser.add_argument("--num_steps", type=int, default=10, help="Measurement steps.")
    
    # Data Arguments
    parser.add_argument("--context_length", type=int, default=256, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    
    # System Arguments
    parser.add_argument("--use_bf16", action="store_true", help="Enable bfloat16 autocast.")
    parser.add_argument("--record_memory", action="store_true", help="Enable CUDA memory recording.")
    parser.add_argument("--snapshot_dir", type=str, default="snapshots", 
                        help="Directory to save memory snapshots.")

    args = parser.parse_args()
    
    runner = BenchmarkRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
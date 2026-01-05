import torch
import triton
# Make sure these imports match your file structure
from cs336_systems.triton_attn import flash_fwd_kernel
from cs336_systems.torch_attn import TorchAttention

def test_kernel_directly():
    # 1. Setup Inputs
    BATCH, N_HEADS = 1, 1 
    N_Q, N_K, D = 128, 128, 64
    Q_TILE_SIZE = 128
    K_TILE_SIZE = 64
    
    device = "cuda"
    # Flash Attention is typically designed for float32. 
    # Using float32 is fine for debugging logic, but ensure your kernel handles it.
    dtype = torch.float32 
    
    # Create random inputs
    Q = torch.randn((BATCH, N_Q, D), device=device, dtype=dtype).requires_grad_(True)
    K = torch.randn((BATCH, N_K, D), device=device, dtype=dtype).requires_grad_(True)
    V = torch.randn((BATCH, N_K, D), device=device, dtype=dtype).requires_grad_(True)
    
    # 2. Run Reference Implementation (The "Ground Truth")
    # This was missing in your script!
    print("--- Running Reference (TorchAttention) ---")
    O_ref = TorchAttention.apply(Q, K, V, False) # is_causal=False
    
    # Extract L_ref from saved tensors
    # Order saved in forward: Q, K, V, O, L (index -1)
    L_ref = O_ref.grad_fn.saved_tensors[-1]

    # 3. Setup Triton Outputs
    O_triton = torch.empty_like(Q)
    # L is usually float32 for stability
    L_triton = torch.empty((BATCH, N_Q), device=device, dtype=torch.float32)

    # 4. Launch Kernel
    grid = (triton.cdiv(N_Q, Q_TILE_SIZE), BATCH)
    scale = 1.0 / (D ** 0.5)

    print(f"Launching kernel with grid: {grid}")
    
    flash_fwd_kernel[grid](
        Q, K, V,
        O_triton, L_triton,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O_triton.stride(0), O_triton.stride(1), O_triton.stride(2),
        L_triton.stride(0), L_triton.stride(1),
        N_Q, N_K,
        scale,
        D=D,
        Q_TILE_SIZE=Q_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE,
        is_causal=False
    )

    # 5. Verify correctness
    print("--- Verifying ---")
    
    # Compare O_triton vs O_ref (Not empty O!)
    try:
        torch.testing.assert_close(O_triton, O_ref, rtol=1e-2, atol=1e-2)
        print("✅ Output (O) Matches!")
    except AssertionError as e:
        print("❌ Output (O) Mismatch")
        print(f"   Max Diff: {(O_triton - O_ref).abs().max().item()}")
        
    # Compare L_triton vs L_ref
    try:
        torch.testing.assert_close(L_triton, L_ref, rtol=1e-2, atol=1e-2)
        print("✅ LogSumExp (L) Matches!")
    except AssertionError as e:
        print("❌ LogSumExp (L) Mismatch")
        print(f"   Max Diff: {(L_triton - L_ref).abs().max().item()}")

if __name__ == "__main__":
    test_kernel_directly()

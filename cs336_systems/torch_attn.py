import torch
import math
from einops import rearrange, einsum

class TorchAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: ... q d
        # K: ... k d
        # V: ... k d
        
        # Calculate scale factor
        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # 1. Compute Scores (S)
        # Pattern: queries (q) and keys (k) interact over dimension (d)
        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale

        # 3. Compute LogSumExp (L) for numerical stability and backward pass
        # Reduce over keys (k)
        L = torch.logsumexp(S, dim=-1)

        # 4. Compute Probabilities (P)
        # Expand L to match S: ... q -> ... q 1
        P = torch.exp(S - rearrange(L, "... q -> ... q 1"))

        # 5. Compute Output (O)
        # Pattern: probabilities (q, k) weight the values (k, d) -> result (q, d)
        O = einsum(P, V, "... q k, ... k d -> ... q d")

        # SAVE FOR BACKWARD:
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale # Save scale as a python float or scalar

        return O

    @staticmethod
    def backward(ctx, dO):
        # Retrieve saved tensors
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale

        # 1. Compute D (correction term for Softmax derivative)
        # D is the dot product of dO and O, summed over dimension d
        # Shape: ... q
        D = einsum(dO, O, "... q d, ... q d -> ... q")

        # 2. Recompute P (Probabilities) on-the-fly
        # We need S first: S = QK^T
        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
        
        # P = exp(S - L)
        # Note: L must be broadcasted: ... q -> ... q 1
        P = torch.exp(S - rearrange(L, "... q -> ... q 1"))

        # 3. Compute dP (Gradient wrt Probabilities)
        # dP = dO @ V^T
        # Pattern: ... q d, ... k d -> ... q k
        dP = einsum(dO, V, "... q d, ... k d -> ... q k")

        # 4. Compute dS (Gradient wrt Scores)
        # dS = P * (dP - D)
        # D must be broadcasted: ... q -> ... q 1
        dS = P * (dP - rearrange(D, "... q -> ... q 1")) * scale

        # 5. Compute Gradients for Q, K, V
        # dQ = dS @ K
        # Pattern: ... q k, ... k d -> ... q d
        dQ = einsum(dS, K, "... q k, ... k d -> ... q d")

        # dK = dS^T @ Q
        # Pattern: ... q k, ... q d -> ... k d
        dK = einsum(dS, Q, "... q k, ... q d -> ... k d")

        # dV = P^T @ dO
        # Pattern: ... q k, ... q d -> ... k d
        dV = einsum(P, dO, "... q k, ... q d -> ... k d")

        return dQ, dK, dV, None

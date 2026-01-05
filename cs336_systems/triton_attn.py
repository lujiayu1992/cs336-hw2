import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 1. Unpack shapes directly (Avoids redundant b, q, k, d vars)
        BATCH, N_Q, D = Q.shape
        _, N_K, _ = K.shape

        # 2. Allocate Output & Buffer
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((BATCH, N_Q), device=Q.device, dtype=torch.float32)

        # 3. Constants
        scale = D ** -0.5
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 64
        
        # 4. Kernel Launch
        grid = (triton.cdiv(N_Q, Q_TILE_SIZE), BATCH)
        
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            *Q.stride(), # Unpacks to (stride_qb, stride_qq, stride_qd)
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(), # Unpacks to (stride_lb, stride_lq)
            N_Q, N_K,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # 5. Save for Backward & Return
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.grid = grid
        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        
        return O

    
    @staticmethod
    def backward(ctx, dO):
        pass

@triton.jit()
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # make input block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0 * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0 * K_TILE_SIZE, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq,),
        offsets = (query_tile_index* Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,), # major to minor
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # m_i is initialized to negative infinity so the first max update works correctly
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    
    # l_i and O_i start at zero
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1))
        V = tl.load(V_block_ptr, boundary_check=(0, 1))
        S = tl.dot(Q, tl.trans(K)) * scale

        offs_n = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        mask = offs_n[None, :] < N_KEYS

        if is_causal:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])

        S = tl.where(mask, S, float("-inf"))

        S_row_max = tl.max(S, axis=-1)
        m_new = tl.maximum(m_i, S_row_max)
        P = tl.exp(S-m_new[:, None])
        alpha = m_i - m_new
        l_new = tl.exp(alpha) *  l_i + tl.sum(P, axis= -1)
        O_i = tl.exp(alpha)[:, None] * O_i + tl.dot(P.to(V.dtype), V)
        m_i = m_new
        l_i = l_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    tl.store(O_block_ptr, (1/l_i)[:, None] * O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))


@triton.jit()
def flash_bwd_kernel(
    Q_ptr, O_ptr, dO_ptr,
    K_ptr, V_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    key_tile_index = tl.program_id(0) # first dim from pytorch launch
    batch_index = tl.program_id(1) # second dim from kernel launch
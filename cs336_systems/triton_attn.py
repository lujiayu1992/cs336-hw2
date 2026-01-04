import triton
import triton.language as tl
from einops import rearrange, einsum

@triton.jit
def get_block_ptr(
    ptr, 
    stride_b, stride_s, stride_d,
    N_seq, D_dim,
    batch_idx, 
    tile_idx, 
    BLOCK_SIZE, 
    order=(1, 0)
):
    return tl.make_block_ptr(
        ptr + batch_idx * stride_b,
        shape=(N_seq, D_dim),
        strides=(stride_s, stride_d),
        offsets=(tile_idx * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, D_dim),
        order=order
    )

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        pass
    
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
        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
        


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
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin vendor: regular GEMM + separate epilogue reading the
# native (already regular) strides of k/beta/g directly - no layout
# materialization copies in the wrapper (T28 E11 / T37 E4 / T47
# recipe; E8 proved this family passes correctness on kunlunxin where
# every fused whole-tile form ends in the compile-worker crash
# family). The Gram matrix is computed once per k-group (GQA) in a
# single-trip fp32 IEEE GEMM; per-head scaling / decay / masking lives
# in a second small-tile kernel.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _kkt_gram_gemm_kernel(
    k_ptr,
    gram_ptr,
    M,
    nchunks,
    num_k_heads,
    k_sb,
    k_st,
    k_sh,
    k_sk,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_matrix = num_pid_m * num_pid_n
    matrix_id = pid // tiles_per_matrix
    local = pid - matrix_id * tiles_per_matrix
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = local // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (local % group_size_m)
    pid_n = (local % num_pid_in_group) // group_size_m

    # matrix_id -> (batch, chunk, k-group); all remaining addressing is
    # plain regular strides of the native [B, T, Hg, K] layout.
    kg = matrix_id % num_k_heads
    bc = matrix_id // num_k_heads
    c = bc % nchunks
    bi = bc // nchunks
    k_base = k_ptr + bi * k_sb + kg * k_sh + c * M * k_st

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = k_base + offs_m[:, None] * k_st + offs_k[None, :] * k_sk
    b_ptrs = k_base + offs_k[:, None] * k_sk + offs_n[None, :] * k_st
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Single trip by construction (BLOCK_K = next_pow2(K)): the
    # multi-trip K loop miscompiles on this backend family.
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        # E10: fp16/bf16 operands dot natively (SDNN path; bf16 inputs
        # with fp32 accumulation equal fp32-ieee math on bf16-valued
        # data); true-fp32 inputs stay ieee fp32.
        if not USE_INPUT_DTYPE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * k_sk
        b_ptrs += BLOCK_K * k_sk

    gram_ptrs = (
        gram_ptr
        + matrix_id * M * N
        + offs_m[:, None] * N
        + offs_n[None, :]
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(gram_ptrs, accumulator, mask=mask)


@triton.jit
def _kkt_epilogue_kernel(
    gram_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    seqlen,
    nchunks,
    num_heads,
    num_k_heads,
    ratio,
    beta_sb,
    beta_st,
    beta_sh,
    g_sb,
    g_st,
    g_sh,
    BT: tl.constexpr,
    HAS_G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(BT, BLOCK_M)
    num_pid_n = tl.cdiv(BT, BLOCK_N)
    tiles_per_head = num_pid_m * num_pid_n
    qh = pid // tiles_per_head
    local = pid - qh * tiles_per_head
    m_tile = local // num_pid_n
    n_tile = local - m_tile * num_pid_n

    h = qh % num_heads
    bc = qh // num_heads
    c = bc % nchunks
    b = bc // nchunks
    kg = h // ratio
    gram_id = bc * num_k_heads + kg

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_mn = (offs_m[:, None] < BT) & (offs_n[None, :] < BT)

    gram_base = gram_ptr + gram_id * BT * BT
    result = tl.load(
        gram_base + offs_m[:, None] * BT + offs_n[None, :],
        mask=mask_mn,
        other=0.0,
    )

    # Same scaling order as the generic kernel: beta first, then the
    # safe-exp decay, then the strict lower-triangular zeroing.
    beta_base = beta_ptr + b * beta_sb + h * beta_sh + c * BT * beta_st
    beta_m = tl.load(
        beta_base + offs_m * beta_st, mask=offs_m < BT, other=0.0
    ).to(tl.float32)
    result = result * beta_m[:, None]
    if HAS_G:
        g_base = g_ptr + b * g_sb + h * g_sh + c * BT * g_st
        g_m = tl.load(
            g_base + offs_m * g_st, mask=offs_m < BT, other=0.0
        ).to(tl.float32)
        g_n = tl.load(
            g_base + offs_n * g_st, mask=offs_n < BT, other=0.0
        ).to(tl.float32)
        g_diff = g_m[:, None] - g_n[None, :]
        result = result * tl.where(g_diff <= 0.0, tl.exp(g_diff), 0.0)
    result = tl.where(offs_m[:, None] > offs_n[None, :], result, 0.0)

    # Output is allocated contiguous [B, T, H, BT].
    out_off = (
        ((b * seqlen + c * BT + offs_m[:, None]) * num_heads + h) * BT
        + offs_n[None, :]
    )
    tl.store(out_ptr + out_off, result, mask=mask_mn)


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    batch, seqlen, num_k_heads, k_size = k.shape
    num_heads = beta.shape[-1]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")
    if num_heads % num_k_heads:
        raise ValueError("num_heads must be divisible by num_k_heads")
    ratio = num_heads // num_k_heads
    nchunks = seqlen // chunk_size
    bt = chunk_size
    output = torch.empty(
        (batch, seqlen, num_heads, bt),
        dtype=torch.float32,
        device=k.device,
    )
    if output.numel() == 0:
        return output
    has_g = g_cumsum is not None and g_cumsum is not beta
    if not has_g:
        g_cumsum = beta

    q_count = batch * nchunks * num_k_heads
    gram = torch.empty(
        (q_count, bt, bt), dtype=torch.float32, device=k.device
    )

    # Stage 1: one flattened regular fp32 IEEE GEMM over all Q Gram
    # matrices, reading k's native regular strides directly.
    block_k = min(triton.next_power_of_2(max(k_size, 16)), 512)
    tiles = triton.cdiv(bt, 32) * triton.cdiv(bt, 32)
    _kkt_gram_gemm_kernel[(q_count * tiles,)](
        k,
        gram,
        bt,
        nchunks,
        num_k_heads,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        N=bt,
        K=k_size,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=block_k,
        GROUP_M=8,
        USE_INPUT_DTYPE=k.dtype in (torch.float16, torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    # Stage 2: per-head epilogue (beta / decay / strict lower mask)
    # reading beta/g natively and writing the contiguous output.
    qh_count = batch * nchunks * num_heads
    e_m, e_n = 16, 32
    e_tiles = triton.cdiv(bt, e_m) * triton.cdiv(bt, e_n)
    _kkt_epilogue_kernel[(qh_count * e_tiles,)](
        gram,
        beta,
        g_cumsum,
        output,
        seqlen,
        nchunks,
        num_heads,
        num_k_heads,
        ratio,
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        g_cumsum.stride(0),
        g_cumsum.stride(1),
        g_cumsum.stride(2),
        BT=bt,
        HAS_G=has_g,
        BLOCK_M=e_m,
        BLOCK_N=e_n,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

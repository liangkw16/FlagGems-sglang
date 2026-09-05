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

# Kunlunxin vendor: layout materialization + regular GEMM + separate
# epilogue (T28 E11 / T37 E4 / T47 recipe). The fused whole-tile kernel
# family (direct dot, block_ptr + tl.trans, FLA persistent) all end in
# the compile-worker crash family on this chip; the only proven-working
# form is a completely regular single-trip fp32 IEEE GEMM with no
# indirect indexing and no runtime branches. The Gram matrix is
# computed once per k-group (GQA) and the per-head scaling / decay /
# masking lives in a second small-tile kernel.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _kkt_gram_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    a_matrix_stride,
    stride_am,
    stride_ak,
    b_matrix_stride,
    stride_bk,
    stride_bn,
    c_matrix_stride,
    stride_cm,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + matrix_id * a_matrix_stride
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + matrix_id * b_matrix_stride
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
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
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + matrix_id * c_matrix_stride
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


@triton.jit
def _kkt_epilogue_kernel(
    gram_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    num_heads,
    num_k_heads,
    ratio,
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

    outer = qh // num_heads
    h = qh - outer * num_heads
    kg = h // ratio
    gram_id = outer * num_k_heads + kg

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
    beta_m = tl.load(beta_ptr + qh * BT + offs_m, mask=offs_m < BT, other=0.0)
    result = result * beta_m[:, None]
    if HAS_G:
        g_m = tl.load(g_ptr + qh * BT + offs_m, mask=offs_m < BT, other=0.0)
        g_n = tl.load(g_ptr + qh * BT + offs_n, mask=offs_n < BT, other=0.0)
        g_diff = g_m[:, None] - g_n[None, :]
        result = result * tl.where(g_diff <= 0.0, tl.exp(g_diff), 0.0)
    result = tl.where(offs_m[:, None] > offs_n[None, :], result, 0.0)

    out_off = (
        (outer * BT + offs_m[:, None]) * (num_heads * BT)
        + h * BT
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

    # Stage 0: materialize regular [Q, BT, K] / [QH, BT] layouts so the
    # GEMM kernel never sees indirect indexing or broadcast strides.
    q_count = batch * nchunks * num_k_heads
    qh_count = batch * nchunks * num_heads
    k_m = (
        k.reshape(batch, nchunks, bt, num_k_heads, k_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(q_count, bt, k_size)
        .float()
    )
    beta_m = (
        beta.reshape(batch, nchunks, bt, num_heads)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(qh_count, bt)
        .float()
    )
    if has_g:
        g_m = (
            g_cumsum.reshape(batch, nchunks, bt, num_heads)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(qh_count, bt)
            .float()
        )
    else:
        g_m = beta_m

    # Stage 1: one flattened regular fp32 IEEE GEMM over all Q Gram
    # matrices (single-trip K by construction).
    gram = torch.empty(
        (q_count, bt, bt), dtype=torch.float32, device=k.device
    )
    block_k = min(triton.next_power_of_2(max(k_size, 16)), 512)
    tiles = triton.cdiv(bt, 32) * triton.cdiv(bt, 32)
    _kkt_gram_gemm_kernel[(q_count * tiles,)](
        k_m,
        k_m,
        gram,
        bt,
        k_m.stride(0),
        k_m.stride(1),
        k_m.stride(2),
        k_m.stride(0),
        k_m.stride(2),
        k_m.stride(1),
        gram.stride(0),
        gram.stride(1),
        gram.stride(2),
        N=bt,
        K=k_size,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=1,
    )

    # Stage 2: per-head epilogue (beta / decay / strict lower mask)
    # writing straight into the final contiguous output.
    e_m, e_n = 16, 32
    e_tiles = triton.cdiv(bt, e_m) * triton.cdiv(bt, e_n)
    _kkt_epilogue_kernel[(qh_count * e_tiles,)](
        gram,
        beta_m,
        g_m,
        output,
        num_heads,
        num_k_heads,
        ratio,
        BT=bt,
        HAS_G=has_g,
        BLOCK_M=e_m,
        BLOCK_N=e_n,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

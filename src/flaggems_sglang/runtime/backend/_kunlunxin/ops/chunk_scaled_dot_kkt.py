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
    rows,
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
    HBT: tl.constexpr,
    HAS_G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # E13: row-tiled epilogue - one program per output row (b, s) with
    # the row decomposition in SCALAR math and only ONE vector division
    # per lane (h = lane // BT; n recovered by mul-sub). The E11 flat
    # form chained four per-lane divisions and sits at 0.048x.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for r in range(pid, rows, nprog):
        s = r % seqlen
        b = r // seqlen
        c = s // BT
        m = s - c * BT
        bc = b * nchunks + c
        row_base = r.to(tl.int64) * HBT
        for l0 in range(0, HBT, BLOCK):
            lane = l0 + tl.arange(0, BLOCK)
            lmask = lane < HBT
            h = lane // BT
            n = lane - h * BT
            kg = h // ratio
            gram_id = bc * num_k_heads + kg

            result = tl.load(
                gram_ptr + gram_id * (BT * BT) + m * BT + n,
                mask=lmask,
                other=0.0,
            )
            # Same scaling order as the generic kernel: beta first,
            # then the safe-exp decay, then the strict lower zeroing.
            beta_m = tl.load(
                beta_ptr + b * beta_sb + s * beta_st + h * beta_sh,
                mask=lmask,
                other=0.0,
            ).to(tl.float32)
            result = result * beta_m
            if HAS_G:
                g_base = g_ptr + b * g_sb + h * g_sh
                g_m = tl.load(
                    g_base + s * g_st, mask=lmask, other=0.0
                ).to(tl.float32)
                g_n = tl.load(
                    g_base + (c * BT + n) * g_st, mask=lmask, other=0.0
                ).to(tl.float32)
                g_diff = g_m - g_n
                # E14: exp2 with the log2(e) prefactor - if the FlagTree
                # exp lowering is a slow libm path while exp2 is native,
                # the per-element decay dominates the epilogue at 0.063x.
                result = result * tl.where(
                    g_diff <= 0.0,
                    tl.math.exp2(g_diff * 1.4426950408889634),
                    0.0,
                )
            result = tl.where(m > n, result, 0.0)
            tl.store(out_ptr + row_base + lane, result, mask=lmask)


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
    tiles = triton.cdiv(bt, 64) * triton.cdiv(bt, 64)
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=block_k,
        GROUP_M=8,
        USE_INPUT_DTYPE=k.dtype in (torch.float16, torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    # Stage 2: row-tiled epilogue (beta / decay / strict lower mask)
    # reading beta/g natively and writing the contiguous output.
    rows = batch * seqlen
    hbt = num_heads * bt
    grid2 = (min(rows, 65535),)
    _kkt_epilogue_kernel[grid2](
        gram,
        beta,
        g_cumsum,
        output,
        rows,
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
        HBT=hbt,
        HAS_G=has_g,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

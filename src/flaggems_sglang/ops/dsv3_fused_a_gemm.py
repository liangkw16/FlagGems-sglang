# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Triton port of SGLang 8014d9d dsv3_fused_a_gemm (SM90 skinny-M GEMM).

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv3_fused_a_gemm(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    as0,
    bs0,
    bs1,
    os0,
    DOT_IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, tl.cdiv(N, BLOCK_N), tl.num_programs(0)):
        offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N
        offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            a = tl.load(
                a_ptr + offs_m[:, None] * as0 + offs_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * bs0 + offs_n[None, :] * bs1,
                mask=mask_n[None, :],
                other=0.0,
            )
            if DOT_IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
        tl.store(
            out_ptr + offs_m[:, None] * os0 + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _dsv3_fused_a_gemm_baseline(mat_a, mat_b):
    assert mat_a.ndim == 2 and mat_b.ndim == 2
    m, k = mat_a.shape
    n = mat_b.shape[1]
    assert mat_a.shape[1] == mat_b.shape[0]
    assert mat_a.stride(1) == 1
    assert 1 <= m <= 16
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    if m and n and k:
        _dsv3_fused_a_gemm[(min(triton.cdiv(n, 64), 65535),)](
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            mat_a.stride(0),
            mat_b.stride(0),
            mat_b.stride(1),
            out.stride(0),
            DOT_IEEE=mat_a.dtype == torch.float32,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=128,
            num_warps=4,
            # Deeper K pipeline per the FlagGems M=16 tune table (top
            # configs all run num_stages>=4); the ieee path is proxy-only
            # coverage and its fp32 tiles overflow shared memory at 4.
            num_stages=2 if mat_a.dtype == torch.float32 else 4,
        )
    return out


@triton.jit
def _dsv3_fused_a_gemm_splitk(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    as0,
    bs0,
    bs1,
    os0,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Reuse the split-task mapping and fp32 partial layout of T27 d44c85b.
    for task in range(
        tl.program_id(0), tl.cdiv(N, BLOCK_N) * SPLITS, tl.num_programs(0)
    ):
        split = task % SPLITS
        block = task // SPLITS
        k_chunk = tl.cdiv(K, SPLITS * BLOCK_K) * BLOCK_K
        k_begin = split * k_chunk
        k_end = tl.minimum(k_begin + k_chunk, K)
        offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = offs_n < N
        offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(k_begin, k_end, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            a = tl.load(
                a_ptr + offs_m[:, None] * as0 + offs_k[None, :],
                mask=mask_m[:, None] & (offs_k[None, :] < k_end),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * bs0 + offs_n[None, :] * bs1,
                mask=mask_n[None, :] & (offs_k[:, None] < k_end),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
        tl.store(
            out_ptr + split * M * N + offs_m[:, None] * os0 + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _dsv3_reduce_splitk(partials, out, size, BLOCK: tl.constexpr):
    for tile in range(
        tl.program_id(0), tl.cdiv(size, BLOCK), tl.num_programs(0)
    ):
        offs = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < size
        # T27's deterministic fp32 partial sum; no atomics or early cast.
        value = tl.load(partials + offs, mask, other=0.0)
        for split in range(1, 4):
            value += tl.load(partials + split * size + offs, mask, other=0.0)
        tl.store(out + offs, value.to(out.dtype.element_ty), mask)


def dsv3_fused_a_gemm(mat_a, mat_b):
    assert mat_a.ndim == 2 and mat_b.ndim == 2
    m, k = mat_a.shape
    n = mat_b.shape[1]
    assert k == mat_b.shape[0]
    assert mat_a.stride(1) == 1
    assert 1 <= m <= 16
    # Small K cannot amortize a second launch; the strict fp32 proxy path
    # keeps the exact existing IEEE kernel and accumulation order.
    if (
        k < 4096
        or n > 512
        or mat_a.dtype not in (torch.bfloat16, torch.float16)
        or not n
    ):
        return _dsv3_fused_a_gemm_baseline(mat_a, mat_b)
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    partials = torch.empty((4, m, n), dtype=torch.float32, device=mat_a.device)
    _dsv3_fused_a_gemm_splitk[(min(triton.cdiv(n, 64) * 4, 65535),)](
        mat_a,
        mat_b,
        partials,
        m,
        n,
        k,
        mat_a.stride(0),
        mat_b.stride(0),
        mat_b.stride(1),
        n,
        SPLITS=4,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=128,
        num_warps=4,
        num_stages=4,
    )
    _dsv3_reduce_splitk[(min(triton.cdiv(m * n, 256), 65535),)](
        partials, out, m * n, BLOCK=256, num_warps=4
    )
    return out


__all__ = ["dsv3_fused_a_gemm"]

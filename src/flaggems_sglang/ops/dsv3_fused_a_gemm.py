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


def dsv3_fused_a_gemm(mat_a, mat_b):
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
            num_stages=2,
        )
    return out


__all__ = ["dsv3_fused_a_gemm"]

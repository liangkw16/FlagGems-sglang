# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 106 tiny_n_gemm: decode-stage skinny-M bf16 GEMM out = x @ w.T
# with m <= 16. One program per N block: the full [16, BK] x tile (M
# padded to tl.dot's minimum, masked) is multiplied against a [BN, BK]
# w tile with fp32 accumulation - numerically the reference's fp32
# matmul of bf16-representable values. BLOCK_K is capped at 128 so a
# single-trip next_pow2(k) tile never blows shared memory (k=4096
# demanded 1.3MB); larger k takes the masked multi-trip loop.

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny_n_gemm_kernel(
    x,
    w,
    out,
    M,
    N,
    K,
    xs0,
    xs1,
    ws0,
    ws1,
    os0,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = pid.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mm = rm < M
    nm = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        km = rk < K
        xv = tl.load(
            x + rm[:, None] * xs0 + rk[None, :] * xs1,
            mask=mm[:, None] & km[None, :],
            other=0,
        )
        wv = tl.load(
            w + rn[:, None] * ws0 + rk[None, :] * ws1,
            mask=nm[:, None] & km[None, :],
            other=0,
        )
        acc += tl.dot(xv, tl.trans(wv), out_dtype=tl.float32)
    tl.store(
        out + rm[:, None] * os0 + rn[None, :],
        acc.to(out.dtype.element_ty),
        mask=mm[:, None] & nm[None, :],
    )


def tiny_n_gemm(x, w, out_dtype):
    assert x.dim() == 2 and w.dim() == 2
    m, k = x.shape
    n = w.shape[0]
    assert m <= 16
    assert x.dtype == w.dtype == torch.bfloat16
    out = (
        torch.zeros((m, n), dtype=out_dtype, device=x.device)
        if k == 0
        else torch.empty((m, n), dtype=out_dtype, device=x.device)
    )
    if m and n and k:
        _tiny_n_gemm_kernel[(triton.cdiv(n, 64),)](
            x,
            w,
            out,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=min(128, max(16, triton.next_power_of_2(k))),
            num_warps=4,
        )
    return out


__all__ = ["tiny_n_gemm"]

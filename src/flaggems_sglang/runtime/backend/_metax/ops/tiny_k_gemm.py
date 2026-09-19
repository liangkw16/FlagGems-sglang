# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for tiny_k_gemm: BLOCK_N 64->32 - the K=256 whole-K dot
# tile needs 86KB shared memory at BLOCK_N 64, over muxi's 64KB limit
# (submission 18271 OutOfResources); 32 fits with the same structure.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["m", "n"])
def _tiny_k_gemm(
    x, w, out, m, n, xs0, ws0, os0,
    K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    for nb in range(pid, tl.cdiv(n, BLOCK_N), tl.num_programs(0)):
        rm = tl.arange(0, 16)
        rk = tl.arange(0, K)
        rn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        xm = rm < m
        xn = rn < n
        xt = tl.load(
            x + rm[:, None] * xs0 + rk[None, :],
            xm[:, None],
            other=0.0,
        )
        # w is [n, k] row-major; the dot needs [K, N] so load transposed
        wt = tl.load(
            w + rn[None, :] * ws0 + rk[:, None],
            xn[None, :],
            other=0.0,
        )
        acc = tl.dot(xt, wt, out_dtype=tl.float32)
        tl.store(
            out + rm[:, None] * os0 + rn[None, :],
            acc.to(out.dtype.element_ty),
            xm[:, None] & xn[None, :],
        )


def tiny_k_gemm(x, w, out_dtype):
    assert x.ndim == 2 and w.ndim == 2
    m, k = x.shape
    n = w.shape[0]
    assert w.shape[1] == k and m <= 16
    assert k // 8 in (16, 32)
    assert x.dtype == w.dtype == torch.bfloat16
    assert out_dtype in (torch.bfloat16, torch.float32)
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m and n:
        _tiny_k_gemm[(min(triton.cdiv(n, 32), 2048),)](
            x,
            w,
            out,
            m,
            n,
            x.stride(0),
            w.stride(0),
            out.stride(0),
            K=k,
            BLOCK_N=32,
        )
    return out


__all__ = ["tiny_k_gemm"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 103 recompute_w_u: chunked Gated DeltaNet w/u recomputation.
# Per (batch, chunk, query-head) program loads the [BT, BT] inverse UT
# matrix A and applies it to the beta-scaled value block and the
# gate-scaled GQA key block via two tl.dot calls; k uses the GQA head
# map hk = h // (H // Hg). fp32 accumulators throughout; outputs stored
# back to bf16 with one rounding.

import torch
import triton
import triton.language as tl


@triton.jit
def _recompute_w_kernel(
    k,
    beta,
    g_cumsum,
    A,
    w,
    T,
    H,
    Hg,
    K,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    # split kernels: two tl.dot calls in one program miscompiled the
    # second dot's masked tail on triton 3.7.1 (same codegen family as
    # the platform's multi-trip K-loop hard fact), so w and u each get
    # a single-dot kernel
    pid = tl.program_id(0).to(tl.int64)
    n_chunks = T // BT
    h = pid % H
    hc = pid // H
    c = hc % n_chunks
    b = hc // n_chunks
    hk = h // (H // Hg)

    bh_base = (b * T + c * BT) * H + h
    bvec = tl.load(beta + bh_base + tl.arange(0, BT) * H).to(tl.float32)
    gvec = tl.load(g_cumsum + bh_base + tl.arange(0, BT) * H).to(tl.float32)

    arows = tl.arange(0, BT).to(tl.int64)
    a_base = ((b * T + c * BT) * H + h) * BT
    a_tile = tl.load(
        A + a_base + arows[:, None] * (H * BT) + tl.arange(0, BT)[None, :]
    )
    k_base = ((b * T + c * BT) * Hg + hk) * K
    w_base = ((b * T + c * BT) * H + h) * K

    ko = tl.arange(0, BK).to(tl.int64)
    for k0 in range(0, K, BK):
        kk = k0 + ko
        km = kk < K
        kvec = tl.load(
            k + k_base + arows[:, None] * (Hg * K) + kk[None, :],
            mask=km[None, :],
            other=0,
        ).to(tl.float32)
        scaled = (kvec * (bvec * tl.exp(gvec))[:, None]).to(tl.bfloat16)
        acc = tl.dot(a_tile, scaled, out_dtype=tl.float32)
        tl.store(
            w + w_base + arows[:, None] * (H * K) + kk[None, :],
            acc.to(w.dtype.element_ty),
            mask=km[None, :],
        )


@triton.jit
def _recompute_u_kernel(
    v,
    beta,
    A,
    u,
    T,
    H,
    V,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_chunks = T // BT
    h = pid % H
    hc = pid // H
    c = hc % n_chunks
    b = hc // n_chunks

    bh_base = (b * T + c * BT) * H + h
    bvec = tl.load(beta + bh_base + tl.arange(0, BT) * H).to(tl.float32)

    arows = tl.arange(0, BT).to(tl.int64)
    a_base = ((b * T + c * BT) * H + h) * BT
    a_tile = tl.load(
        A + a_base + arows[:, None] * (H * BT) + tl.arange(0, BT)[None, :]
    )
    v_base = ((b * T + c * BT) * H + h) * V

    vo = tl.arange(0, BV).to(tl.int64)
    for v0 in range(0, V, BV):
        vv = v0 + vo
        vm = vv < V
        vvec = tl.load(
            v + v_base + arows[:, None] * (H * V) + vv[None, :],
            mask=vm[None, :],
            other=0,
        ).to(tl.float32)
        scaled = (vvec * bvec[:, None]).to(tl.bfloat16)
        acc = tl.dot(a_tile, scaled, out_dtype=tl.float32)
        tl.store(
            u + v_base + arows[:, None] * (H * V) + vv[None, :],
            acc.to(u.dtype.element_ty),
            mask=vm[None, :],
        )


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None, "varlen path is out of scope"
    B, T, Hg, K = k.shape
    _, _, H, V = v.shape
    BT = A.shape[-1]
    assert T % BT == 0
    w = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)
    u = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)
    if T:
        grid = (B * (T // BT) * H,)
        _recompute_w_kernel[grid](
            k,
            beta,
            g_cumsum,
            A,
            w,
            T,
            H,
            Hg,
            K,
            BT=BT,
            BK=max(16, triton.next_power_of_2(K)),
            num_warps=4,
        )
        _recompute_u_kernel[grid](
            v,
            beta,
            A,
            u,
            T,
            H,
            V,
            BT=BT,
            BV=max(16, triton.next_power_of_2(V)),
            num_warps=4,
        )
    return w, u


__all__ = ["recompute_w_u"]

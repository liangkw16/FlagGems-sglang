# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Kunlunxin vendor for Task 103 recompute_w_u.

Platform history on this chip: the generic bf16 tl.dot form missed
the tolerance by one near-zero element (0.0195 vs 0.015 allowed -
summation-order error in the cancellation regime), and an IEEE fp32
tl.dot variant hit a PassManager compile wall on every backend. This
vendor avoids tl.dot entirely: an outer-product FMA loop with an
fp64 accumulator. bf16 operands round exactly as the reference does,
each product of bf16 values is exact in fp64, and the fp64 sum makes
the near-zero cancellation error negligible. The op's kunlunxin
headroom (11.4x against a python-loop baseline at s0) absorbs the
slower fp64 path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _recompute_wu_exact(
    k,
    v,
    beta,
    g_cumsum,
    A,
    w,
    u,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_chunks = T // BT
    h = pid % H
    hc = pid // H
    c = hc % n_chunks
    b = hc // n_chunks
    hk = h // (H // Hg)

    bh_base = (b * T + c * BT) * H + h
    arows = tl.arange(0, BT).to(tl.int64)
    a_base = ((b * T + c * BT) * H + h) * BT
    k_base = ((b * T + c * BT) * Hg + hk) * K
    v_base = ((b * T + c * BT) * H + h) * V
    w_base = ((b * T + c * BT) * H + h) * K

    ko = tl.arange(0, BK).to(tl.int64)
    for k0 in range(0, K, BK):
        kk = k0 + ko
        km = kk < K
        acc = tl.zeros((BT, BK), dtype=tl.float64)
        for col in tl.static_range(0, BT):
            acol = tl.load(
                A + a_base + arows * (H * BT) + col,
            ).to(tl.float64)
            krow = tl.load(
                k + k_base + col * (Hg * K) + kk,
                mask=km,
                other=0,
            ).to(tl.float32)
            bscale = tl.load(beta + bh_base + col * H).to(tl.float32)
            gscale = tl.load(g_cumsum + bh_base + col * H).to(tl.float32)
            prod = (krow * bscale * tl.exp(gscale)).to(tl.bfloat16).to(
                tl.float64
            )
            acc += acol[:, None] * prod[None, :]
        tl.store(
            w + w_base + arows[:, None] * (H * K) + kk[None, :],
            acc.to(w.dtype.element_ty),
            mask=km[None, :],
        )

    vo = tl.arange(0, BV).to(tl.int64)
    for v0 in range(0, V, BV):
        vv = v0 + vo
        vm = vv < V
        acc = tl.zeros((BT, BV), dtype=tl.float64)
        for col in tl.static_range(0, BT):
            acol = tl.load(
                A + a_base + arows * (H * BT) + col,
            ).to(tl.float64)
            vrow = tl.load(
                v + v_base + col * (H * V) + vv,
                mask=vm,
                other=0,
            ).to(tl.float32)
            scale = tl.load(beta + bh_base + col * H).to(tl.float32)
            prod = (vrow * scale).to(tl.bfloat16).to(tl.float64)
            acc += acol[:, None] * prod[None, :]
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
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    beta = beta if beta.is_contiguous() else beta.contiguous()
    g_cumsum = (
        g_cumsum if g_cumsum.is_contiguous() else g_cumsum.contiguous()
    )
    A = A if A.is_contiguous() else A.contiguous()
    w = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)
    u = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)
    if T:
        _recompute_wu_exact[(B * (T // BT) * H,)](
            k,
            v,
            beta,
            g_cumsum,
            A,
            w,
            u,
            T,
            H,
            Hg,
            K,
            V,
            BT=BT,
            BK=max(16, triton.next_power_of_2(K)),
            BV=max(16, triton.next_power_of_2(V)),
            num_warps=4,
        )
    return w, u


__all__ = ["recompute_w_u"]

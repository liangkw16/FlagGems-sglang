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

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_head_kernel(
    x_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    hidden,
    hc_mult,
    norm_eps,
    hc_eps,
    D,
    fn_stride_row,
    BLOCK_H: tl.constexpr,
    HC: tl.constexpr,
):
    t = tl.program_id(0)
    scale = tl.load(scale_ptr).to(tl.float32)

    offs_j = tl.arange(0, HC)
    offs_m = tl.arange(0, HC)
    offs_h = tl.arange(0, BLOCK_H)
    j_ok = offs_j < hc_mult
    m_ok = offs_m < hc_mult

    # Pass 1 over the flattened row: per-lane vector accumulators for the
    # sum of squares and each mix defer every tl.sum to after the hidden
    # loop (the S0 form held a [HC, HC, BLOCK_H] fn tile and ran nested
    # reductions inside the loop). fn is streamed as flat [HC, BLOCK_H]
    # tiles, one per hc_fn input row m.
    sqr_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    mix_acc = tl.zeros((HC, BLOCK_H), dtype=tl.float32)
    for h0 in range(0, hidden, BLOCK_H):
        h_ok = offs_h < hidden - h0
        for m in range(HC):
            r = tl.load(
                x_ptr + t * D + m * hidden + h0 + offs_h,
                mask=h_ok & (m < hc_mult),
                other=0.0,
            ).to(tl.float32)
            sqr_acc += r * r
            fn_jm = tl.load(
                fn_ptr
                + offs_j[:, None] * fn_stride_row
                + m * hidden
                + h0
                + offs_h[None, :],
                mask=j_ok[:, None] & h_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            mix_acc += r[None, :] * fn_jm

    r = 1.0 / tl.sqrt(tl.sum(sqr_acc) / D + norm_eps)
    mixes = tl.sum(mix_acc, axis=1) * r
    base = tl.load(base_ptr + offs_j, mask=j_ok, other=0.0).to(tl.float32)
    pre = 1.0 / (1.0 + tl.exp(-(mixes * scale + base))) + hc_eps

    # Pass 2: weighted fold of the hc_mult axis.
    for h0 in range(0, hidden, BLOCK_H):
        h_ok = offs_h < hidden - h0
        x = tl.load(
            x_ptr + t * D + offs_m[:, None] * hidden + h0 + offs_h[None, :],
            mask=m_ok[:, None] & h_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        y = tl.sum(pre[:, None] * x, axis=0)
        tl.store(
            y_ptr + t * hidden + h0 + offs_h,
            y.to(y_ptr.dtype.element_ty),
            mask=h_ok,
        )


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    T, hc_mult, hidden = x.shape
    y = torch.empty((T, hidden), dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    D = hc_mult * hidden

    hc_pow2 = triton.next_power_of_2(hc_mult)
    # Live state per program is mix_acc plus one streamed [HC, BLOCK_H] fn
    # tile; keep HC * BLOCK_H within 2048 elements (256 at HC=8, 512 at
    # HC=4, 1024 at HC=2).
    block_h = max(min(triton.next_power_of_2(hidden), 2048 // hc_pow2), 32)
    _hc_head_kernel[(T,)](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        hidden,
        hc_mult,
        float(norm_eps),
        float(hc_eps),
        D,
        hc_fn.stride(0),
        BLOCK_H=block_h,
        HC=hc_pow2,
        num_warps=4,
        num_stages=2,
    )
    return y


__all__ = ["hc_head"]

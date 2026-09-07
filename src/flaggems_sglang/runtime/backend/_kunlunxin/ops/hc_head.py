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

# Kunlunxin vendor: three flat, regular kernels instead of the fused
# two-pass generic form. Both platform attempts of the single-kernel
# generic structure (s0 seq26 / e1 submission 10668) ended in the
# kunlunxin compile-worker crash family (1830s timeout + Aborted)
# while other operators completed on the same days, so the prime
# suspect is the deeply nested hidden loop with per-iteration
# unrolled [HC, BLOCK_H] fn tiles stalling this backend's compiler.
# This vendor follows the T28 E11 / T37 E4 / T45 E8 recipe (simple
# regular kernels compile fast and pass where fused forms crash):
#   1. per-row sum-of-squares reduction (flat row kernel),
#   2. per-(token, j) mixes GEMV in plain vector FMA - no tl.dot,
#      matching the T45 finding that fp32 dot lowers to the scalar
#      path on this backend anyway,
#   3. per-(token, h-block) fold that recomputes the HC scalar
#      gates inline and writes the output row.
# Math order per element mirrors the reference (rsqrt after the
# mean, sigmoid of (mixes * scale + base), weighted sum over the
# hc axis in fp32 before the single output cast).

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_sumsq_kernel(
    x_ptr,
    sumsq_ptr,
    D,
    rows,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, rows, nprog):
        base = x_ptr + t.to(tl.int64) * D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK):
            offs = d0 + tl.arange(0, BLOCK)
            v = tl.load(base + offs, mask=offs < D, other=0.0).to(tl.float32)
            acc += v * v
        tl.store(sumsq_ptr + t, tl.sum(acc))


@triton.jit
def _hc_mix_kernel(
    x_ptr,
    fn_ptr,
    mix_ptr,
    D,
    tasks,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for task in range(pid, tasks, nprog):
        j = task % HC
        t = task // HC
        x_base = x_ptr + t.to(tl.int64) * D
        f_base = fn_ptr + j.to(tl.int64) * D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK):
            offs = d0 + tl.arange(0, BLOCK)
            m = offs < D
            xv = tl.load(x_base + offs, mask=m, other=0.0).to(tl.float32)
            fv = tl.load(f_base + offs, mask=m, other=0.0).to(tl.float32)
            acc += xv * fv
        tl.store(mix_ptr + task, tl.sum(acc))


@triton.jit
def _hc_fold_kernel(
    x_ptr,
    mix_ptr,
    sumsq_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    hidden,
    D,
    norm_eps,
    hc_eps,
    tasks,
    hblocks,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    scale = tl.load(scale_ptr).to(tl.float32)
    for task in range(pid, tasks, nprog):
        tb = task // hblocks
        hb = task - tb * hblocks
        row = tb.to(tl.int64) * D
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = offs_h < hidden
        rstd = 1.0 / tl.sqrt(tl.load(sumsq_ptr + tb) / D + norm_eps)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for m in range(HC):
            mix = tl.load(mix_ptr + tb * HC + m).to(tl.float32)
            base = tl.load(base_ptr + m).to(tl.float32)
            pre = 1.0 / (1.0 + tl.exp(-(mix * rstd * scale + base)))
            pre = pre + hc_eps
            xv = tl.load(
                x_ptr + row + m * hidden + offs_h, mask=hmask, other=0.0
            ).to(tl.float32)
            acc += pre * xv
        tl.store(
            y_ptr + tb.to(tl.int64) * hidden + offs_h,
            acc.to(y_ptr.dtype.element_ty),
            mask=hmask,
        )


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    T, hc_mult, hidden = x.shape
    y = torch.empty((T, hidden), dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    D = hc_mult * hidden

    sumsq = torch.empty((T,), dtype=torch.float32, device=x.device)
    mixes = torch.empty((T, hc_mult), dtype=torch.float32, device=x.device)

    _hc_sumsq_kernel[(min(T, 65535),)](
        x,
        sumsq,
        D,
        T,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )
    _hc_mix_kernel[(min(T * hc_mult, 65535),)](
        x,
        hc_fn,
        mixes,
        D,
        T * hc_mult,
        HC=hc_mult,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )
    block_h = 512
    hblocks = triton.cdiv(hidden, block_h)
    _hc_fold_kernel[(min(T * hblocks, 65535),)](
        x,
        mixes,
        sumsq,
        hc_scale,
        hc_base,
        y,
        hidden,
        D,
        float(norm_eps),
        float(hc_eps),
        T * hblocks,
        hblocks,
        HC=hc_mult,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=1,
    )
    return y


__all__ = ["hc_head"]

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

"""Kunlunxin vendor for Task 94 concat_mla_absorb_q.

Platform-measured ladder on this chip (generic 2D tile 0.0044x,
scalar-row 1024-floor 0.0814x, RPP=8 0.0656x, exact-width 0.0278x,
fused single-store 0.0712x, flat+constexpr-div 0.009x) plus the
codex-ask structural analysis: the untried family is per-segment
scheduling - decompose each source's linear index by its own
power-of-two width (shift/mask, no division by the 576 row pitch),
cover multiple rows per program with wide vectors, and keep the outer
d0 coordinate scalar. Programs are split between an A-region and a
B-region inside one launch; each moves a BLOCK-wide contiguous span
of one source half. Non-pow2 widths (or d0 beyond the grid cap) fall
back to the measured-best scalar-row kernel.
"""

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535
_FALLBACK_BLOCK = 1024


@triton.jit
def _concat_rows_kernel(
    a,
    b,
    out,
    n_rows,
    d1: tl.constexpr,
    rows_per_prog,
    a_s0,
    a_s1,
    a_s2,
    b_s0,
    b_s1,
    b_s2,
    a_last,
    b_last,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK).to(tl.int64)
    row0 = pid * rows_per_prog
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            i0 = row // d1
            i1 = row - i0 * d1
            a_base = i0 * a_s0 + i1 * a_s1
            b_base = i0 * b_s0 + i1 * b_s1
            o_base = row * (a_last + b_last)
            for ca in range(0, a_last, BLOCK):
                cc = ca + cols
                ma = cc < a_last
                value = tl.load(a + a_base + cc * a_s2, mask=ma, other=0)
                tl.store(out + o_base + cc, value, mask=ma)
            for cb in range(0, b_last, BLOCK):
                cc = cb + cols
                mb = cc < b_last
                value = tl.load(b + b_base + cc * b_s2, mask=mb, other=0)
                tl.store(out + o_base + a_last + cc, value, mask=mb)


@triton.jit
def _concat_segment_kernel(
    a,
    b,
    out,
    d1,
    a_last,
    b_last,
    a_blks,
    blk_total,
    LOG_A: tl.constexpr,
    LOG_B: tl.constexpr,
    a_s0,
    a_s1,
    a_s2,
    b_s0,
    b_s1,
    b_s2,
    BLOCK: tl.constexpr,
):
    # one 1-D launch: pid -> (i0, j) with a single constexpr divmod.
    # j < a_blks selects an A-half chunk (index decomposed by 2**LOG_A),
    # otherwise a B-half chunk (decomposed by 2**LOG_B). Column masks
    # only guard the last chunk of each region.
    pid = tl.program_id(0).to(tl.int64)
    i0 = pid // blk_total
    j = pid - i0 * blk_total
    offs = tl.arange(0, BLOCK).to(tl.int64)
    if j < a_blks:
        e = j * BLOCK + offs
        i1 = e >> LOG_A
        col = e & (a_last - 1)
        valid = e < (d1 * a_last)
        src = a + i0 * a_s0 + i1 * a_s1 + col * a_s2
        value = tl.load(src, mask=valid, other=0)
        tl.store(
            out + (i0 * d1 + i1) * (a_last + b_last) + col,
            value,
            mask=valid,
        )
    else:
        e = (j - a_blks) * BLOCK + offs
        i1 = e >> LOG_B
        col = e & (b_last - 1)
        valid = e < (d1 * b_last)
        src = b + i0 * b_s0 + i1 * b_s1 + col * b_s2
        value = tl.load(src, mask=valid, other=0)
        tl.store(
            out + (i0 * d1 + i1) * (a_last + b_last) + a_last + col,
            value,
            mask=valid,
        )


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def concat_mla_absorb_q(a, b):
    assert a.dtype == b.dtype
    assert a.dim() == 3 and b.dim() == 3
    assert a.shape[:-1] == b.shape[:-1]
    a_last = a.shape[-1]
    b_last = b.shape[-1]
    d0, d1 = a.shape[0], a.shape[1]
    n_rows = d0 * d1
    out = torch.empty(
        (d0, d1, a_last + b_last), dtype=a.dtype, device=a.device
    )
    if n_rows and (a_last + b_last):
        block = 8192
        if (
            _is_pow2(a_last)
            and _is_pow2(b_last)
            and _is_pow2(d1)
            and a_last <= block
            and b_last <= block
        ):
            a_blks = triton.cdiv(d1 * a_last, block)
            b_blks = triton.cdiv(d1 * b_last, block)
            blk_total = a_blks + b_blks
            if d0 * blk_total <= _MAX_GRID:
                _concat_segment_kernel[(d0 * blk_total,)](
                    a,
                    b,
                    out,
                    d1,
                    a_last,
                    b_last,
                    a_blks,
                    blk_total,
                    a_last.bit_length() - 1,
                    b_last.bit_length() - 1,
                    a.stride(0),
                    a.stride(1),
                    a.stride(2),
                    b.stride(0),
                    b.stride(1),
                    b.stride(2),
                    BLOCK=block,
                    num_warps=8,
                )
                return out
        _concat_rows_kernel[(min(triton.cdiv(n_rows, 1), _MAX_GRID),)](
            a,
            b,
            out,
            n_rows,
            d1,
            triton.cdiv(n_rows, _MAX_GRID),
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            a_last,
            b_last,
            BLOCK=_FALLBACK_BLOCK,
            num_warps=8,
        )
    return out


__all__ = ["concat_mla_absorb_q"]

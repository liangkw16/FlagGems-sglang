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

Platform-measured history on this chip: generic 2D tile 0.0044x, 1D
scalar-row vendor with a 1024-lane floor 0.0814x, RPP=8 packing
0.0656x, exact-width 512/64 segments 0.0278x, fused single-store 1024
vector 0.0712x - scalar-row 1D vectors win, 2D tiles and narrow
vectors lose, and the successful kunlunxin kernels in this batch are
flat and wide (T95 27x, T96 0.583x, T98 0.350x; PR#56 validates
16384-lane flat blocks, PR#69 validates constexpr divisions).

This variant is fully flat over the output tensor: one BLOCK-wide
vector per program covers whole rows at a time; the per-element row
and column come from divisions by JIT-constant widths so they
strength-reduce to multiply-shifts instead of the XPU
software-division path. Launch stays under the 65535-program cap for
any realistic shape (172.8M-element platform case 8 needs 10547).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 16384


@triton.jit
def _concat_flat_kernel(
    a,
    b,
    out,
    numel,
    d1: tl.constexpr,
    width: tl.constexpr,
    a_last: tl.constexpr,
    a_s0,
    a_s1,
    a_s2,
    b_s0,
    b_s1,
    b_s2,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    row = offs // width
    col = offs - row * width
    i0 = row // d1
    i1 = row - i0 * d1
    ma = m & (col < a_last)
    mb = m & (col >= a_last)
    av = tl.load(a + i0 * a_s0 + i1 * a_s1 + col * a_s2, mask=ma, other=0)
    bv = tl.load(
        b + i0 * b_s0 + i1 * b_s1 + (col - a_last) * b_s2, mask=mb, other=0
    )
    tl.store(out + offs, tl.where(col < a_last, av, bv), mask=m)


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
    numel = n_rows * (a_last + b_last)
    if numel:
        _concat_flat_kernel[(triton.cdiv(numel, _BLOCK),)](
            a,
            b,
            out,
            numel,
            d1,
            a_last + b_last,
            a_last,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            BLOCK=_BLOCK,
            num_warps=8,
        )
    return out


__all__ = ["concat_mla_absorb_q"]

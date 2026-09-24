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

The generic 2D row-tile kernel read 0.0044x on the platform's kunlunxin
evaluator while every other chip passed at 1.0-2.2x, so the XPU backend
handles the broadcast 2D tile shape pathologically. This variant keeps
every access a 1D vector op - the shape the same batch's
create_chunked_prefix_cache_kv_indices kernel used to reach 27x on
kunlunxin: one program owns a few consecutive rows, each row is two
masked 1D segment copies through the runtime strides, and the launch
stays under the 65535 program cap. Not compiled on kunlunxin hardware
locally; the platform evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


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
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    acols = tl.arange(0, BLOCK_A).to(tl.int64)
    bcols = tl.arange(0, BLOCK_B).to(tl.int64)
    ma = acols < a_last
    mb = bcols < b_last
    row0 = pid * rows_per_prog
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            i0 = row // d1
            i1 = row - i0 * d1
            a_base = i0 * a_s0 + i1 * a_s1
            b_base = i0 * b_s0 + i1 * b_s1
            o_base = row * (a_last + b_last)
            value = tl.load(a + a_base + acols * a_s2, mask=ma, other=0)
            tl.store(out + o_base + acols, value, mask=ma)
            value = tl.load(b + b_base + bcols * b_s2, mask=mb, other=0)
            tl.store(out + o_base + a_last + bcols, value, mask=mb)


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
        # e3: exact-width segment vectors (BLOCK_A/BLOCK_B) replace the
        # 1024-lane floor that idled half the threads on 576-wide rows;
        # d1 is a JIT constant so the per-row division strength-reduces.
        # e2 showed row packing hurts (0.0656 at RPP=8 vs 0.0814 at 1),
        # so one row per program stays
        block_a = min(65536, max(16, triton.next_power_of_2(a_last)))
        block_b = min(65536, max(16, triton.next_power_of_2(b_last)))
        rows_per_prog = triton.cdiv(n_rows, _MAX_GRID)
        grid = (triton.cdiv(n_rows, rows_per_prog),)
        _concat_rows_kernel[grid](
            a,
            b,
            out,
            n_rows,
            d1,
            rows_per_prog,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            a_last,
            b_last,
            BLOCK_A=block_a,
            BLOCK_B=block_b,
            num_warps=8,
        )
    return out


__all__ = ["concat_mla_absorb_q"]

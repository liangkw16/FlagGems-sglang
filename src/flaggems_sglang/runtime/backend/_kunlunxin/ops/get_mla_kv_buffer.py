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

"""Kunlunxin vendor for Task 98 get_mla_kv_buffer.

The generic 2D row-tile kernel read 0.045x on kunlunxin while every
other chip passed at 0.6-1.8x; the batch's 1D-vector-shaped
kv-indices kernel reached 27x on the same backend, so this variant
moves every access to 1D vector ops: one program owns a few
consecutive loc rows, each row gathers the NoPE and RoPE halves as
two masked 1D segments through the runtime strides (implicit dtype
conversion at store) and the launch stays under 65535 programs.
Not compiled on kunlunxin hardware locally.
"""

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _kv_rows_kernel(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    n_rows,
    rows_per_prog,
    loc_s0,
    kv_s0,
    kv_s1,
    nope_dim,
    rope_dim,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ncols = tl.arange(0, BLOCK_N).to(tl.int64)
    rcols = tl.arange(0, BLOCK_R).to(tl.int64)
    row0 = pid * rows_per_prog
    for r in range(0, rows_per_prog):
        row = row0 + r
        if row < n_rows:
            idx = tl.load(loc + row * loc_s0).to(tl.int64)
            src = idx * kv_s0
            for cn in range(0, nope_dim, BLOCK_N):
                cc = cn + ncols
                mn = cc < nope_dim
                nv = tl.load(kv_buffer + src + cc * kv_s1, mask=mn, other=0)
                tl.store(
                    nope_out + row * nope_dim + cc,
                    nv.to(nope_out.dtype.element_ty),
                    mask=mn,
                )
            for cr in range(0, rope_dim, BLOCK_R):
                cc = cr + rcols
                mr = cc < rope_dim
                rv = tl.load(
                    kv_buffer + src + (nope_dim + cc) * kv_s1, mask=mr, other=0
                )
                tl.store(
                    rope_out + row * rope_dim + cc,
                    rv.to(rope_out.dtype.element_ty),
                    mask=mr,
                )


def get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n = loc.shape[0]
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = kv_buffer.shape[-1] - nope_dim
    nope = torch.empty(
        (n, nope_dim), dtype=cache_k_nope.dtype, device=kv_buffer.device
    )
    rope = torch.empty(
        (n, rope_dim), dtype=cache_k_rope.dtype, device=kv_buffer.device
    )
    if n and (nope_dim or rope_dim):
        # exact-width segments (no 1024 floor): T94's e3 measurement
        # showed the floor idles half the lanes on 576-wide rows and the
        # exact-width vectors recovered kunlunxin throughput
        block_n = min(65536, max(16, triton.next_power_of_2(max(nope_dim, 1))))
        block_r = min(65536, max(16, triton.next_power_of_2(max(rope_dim, 1))))
        rows_per_prog = triton.cdiv(n, _MAX_GRID)
        grid = (triton.cdiv(n, rows_per_prog),)
        _kv_rows_kernel[grid](
            kv_buffer,
            loc,
            nope,
            rope,
            n,
            rows_per_prog,
            loc.stride(0),
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            nope_dim,
            rope_dim,
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            num_warps=8,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]

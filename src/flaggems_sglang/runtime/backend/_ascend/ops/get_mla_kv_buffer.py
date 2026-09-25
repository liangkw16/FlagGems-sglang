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

"""Ascend vendor for Task 98 get_mla_kv_buffer.

Our generic one-shot 2D row-tile kernel read 0.72x on the platform's
Ascend evaluator (leader 1.19); the batch's own T93/T97 measurements
showed the PR#70 platform-validated elementwise recipe - a persistent
grid-stride pass with the CTA count capped at 48 and num_warps=4 -
lifts this op family on that chip (+115% int add, +31% sigmoid gate).
This vendor persistent-izes the gather-split: a small fixed grid
strides over row tiles, each iteration gathering BLOCK_R rows and
streaming both halves through the runtime strides with implicit store
conversion. Not compiled on Ascend hardware locally; the platform
evaluator is the arbiter.
"""

import torch
import triton
import triton.language as tl

_BLOCK_R = 8
_BLOCK_C = 512
_PERSISTENT = 48


@triton.jit
def _get_mla_kv_buffer_persistent(
    kv_buffer,
    loc,
    nope_out,
    rope_out,
    n_rows,
    loc_s0,
    kv_s0,
    kv_s1,
    nope_dim,
    rope_dim,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_C).to(tl.int64)
    rvec = tl.arange(0, BLOCK_R).to(tl.int64)
    for tile in range(tl.program_id(0), tl.cdiv(n_rows, BLOCK_R), tl.num_programs(0)):
        rows = tile * BLOCK_R + rvec
        rmask = rows < n_rows
        idx = tl.load(loc + rows * loc_s0, mask=rmask, other=0).to(tl.int64)
        base = idx * kv_s0
        for c0 in range(0, nope_dim, BLOCK_C):
            cc = c0 + cols
            m = rmask[:, None] & (cc[None, :] < nope_dim)
            value = tl.load(
                kv_buffer + base[:, None] + cc[None, :] * kv_s1, mask=m, other=0
            )
            dst = rows[:, None] * nope_dim + cc[None, :]
            tl.store(
                nope_out + dst, value.to(nope_out.dtype.element_ty), mask=m
            )
        for c0 in range(0, rope_dim, BLOCK_C):
            cc = c0 + cols
            m = rmask[:, None] & (cc[None, :] < rope_dim)
            value = tl.load(
                kv_buffer + base[:, None] + (nope_dim + cc[None, :]) * kv_s1,
                mask=m,
                other=0,
            )
            dst = rows[:, None] * rope_dim + cc[None, :]
            tl.store(
                rope_out + dst, value.to(rope_out.dtype.element_ty), mask=m
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
        grid = (min(triton.cdiv(n, _BLOCK_R), _PERSISTENT),)
        _get_mla_kv_buffer_persistent[grid](
            kv_buffer,
            loc,
            nope,
            rope,
            n,
            loc.stride(0),
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            nope_dim,
            rope_dim,
            BLOCK_R=_BLOCK_R,
            BLOCK_C=_BLOCK_C,
            num_warps=4,
        )
    return nope, rope


__all__ = ["get_mla_kv_buffer"]

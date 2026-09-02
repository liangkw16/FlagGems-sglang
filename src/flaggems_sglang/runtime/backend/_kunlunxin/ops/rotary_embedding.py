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

_MAX_GRID = 65535


@triton.jit
def _rotary_embedding_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    total_rows,
    half_dim,
    HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    # one program per (t, h) row; GPT-J even/odd pair layout (the task
    # reference always splits x[..., 0::2] / x[..., 1::2] regardless of
    # the interleaved flag)
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    i = tl.arange(0, HALF_DIM)
    mask = i < half_dim
    for row in range(pid, total_rows, grid_stride):
        t = row // HEADS
        base = row * (2 * half_dim)
        x_base = x_ptr + base
        o_base = out_ptr + base
        cs_base = t * half_dim + i
        x1 = tl.load(x_base + 2 * i, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + 2 * i + 1, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + cs_base, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_base, mask=mask, other=0.0).to(tl.float32)
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        out_ty = out_ptr.dtype.element_ty
        tl.store(o_base + 2 * i, o1.to(out_ty), mask=mask)
        tl.store(o_base + 2 * i + 1, o2.to(out_ty), mask=mask)


def rotary_embedding(x, cos, sin, interleaved):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    num_tokens, num_heads, dim = x.shape
    half_dim = dim // 2
    output = torch.empty_like(x)
    if num_tokens * num_heads == 0 or half_dim == 0:
        return output
    total_rows = num_tokens * num_heads
    grid = (min(total_rows, _MAX_GRID),)
    _rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        output,
        total_rows,
        half_dim,
        HEADS=num_heads,
        HALF_DIM=triton.next_power_of_2(half_dim),
    )
    return output


__all__ = ["rotary_embedding"]

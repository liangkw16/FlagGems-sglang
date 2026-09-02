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
    FULL: tl.constexpr,
    HALF: tl.constexpr,
):
    # one program per (t, h) row (unchanged); GPT-J even/odd pair layout.
    # e6: the whole row is touched as ONE contiguous vector per program -
    # load + tl.reshape/tl.split deinterleave in registers, tl.join +
    # reshape for a contiguous store - replacing the stride-2 loads and
    # stores of the e3 bytes (0.289x). BLOCK-style grid/one-row structure
    # is kept: it is the only Kunlun-verified axis family.
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    j = tl.arange(0, FULL)
    jmask = j < 2 * half_dim
    i = tl.arange(0, HALF)
    imask = i < half_dim
    for row in range(pid, total_rows, grid_stride):
        t = row // HEADS
        base = row * (2 * half_dim)
        cs_base = t * half_dim + i
        xv = tl.load(x_ptr + base + j, mask=jmask, other=0.0).to(tl.float32)
        x1, x2 = tl.split(tl.reshape(xv, (HALF, 2)))
        c = tl.load(cos_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        out = tl.reshape(tl.join(o1, o2), (FULL,))
        out_ty = out_ptr.dtype.element_ty
        tl.store(out_ptr + base + j, out.to(out_ty), mask=jmask)


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
    half_pow2 = triton.next_power_of_2(half_dim)
    _rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        output,
        total_rows,
        half_dim,
        HEADS=num_heads,
        FULL=2 * half_pow2,
        HALF=half_pow2,
    )
    return output


__all__ = ["rotary_embedding"]

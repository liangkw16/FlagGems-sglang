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
_HEADS_TILE = 8


@triton.jit
def _rotary_embedding_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    total_tiles,
    head_tiles,
    num_heads,
    half_dim,
    HEADS_TILE: tl.constexpr,
    FULL: tl.constexpr,
    HALF: tl.constexpr,
):
    # enflame vendor (e6): the generic e3 tile structure, but every row is
    # accessed as ONE contiguous vector (load + tl.reshape/tl.split
    # deinterleave in registers, tl.join + reshape for a contiguous
    # store). E4 showed the enflame bottleneck is not tile width but the
    # stride-2 even/odd access pattern itself, so this changes only the
    # access pattern; cos/sin reuse over the head tile is unchanged.
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    h_offs = tl.arange(0, HEADS_TILE)
    j = tl.arange(0, FULL)
    jmask = j < 2 * half_dim
    i = tl.arange(0, HALF)
    imask = i < half_dim
    for tile in range(pid, total_tiles, grid_stride):
        t = tile // head_tiles
        head_tile = tile - t * head_tiles
        h0 = head_tile * HEADS_TILE
        hmask = h0 + h_offs < num_heads
        mask2d = hmask[:, None] & jmask[None, :]
        rows = t * num_heads + h0 + h_offs
        offs2d = rows[:, None] * (2 * half_dim) + j[None, :]
        xv = tl.load(x_ptr + offs2d, mask=mask2d, other=0.0).to(tl.float32)
        x1, x2 = tl.split(tl.reshape(xv, (HEADS_TILE, HALF, 2)))
        cs_base = t * half_dim + i
        c = tl.load(cos_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        o1 = x1 * c[None, :] - x2 * s[None, :]
        o2 = x1 * s[None, :] + x2 * c[None, :]
        out = tl.reshape(tl.join(o1, o2), (HEADS_TILE, FULL))
        out_ty = out_ptr.dtype.element_ty
        tl.store(out_ptr + offs2d, out.to(out_ty), mask=mask2d)


def rotary_embedding(x, cos, sin, interleaved):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    num_tokens, num_heads, dim = x.shape
    half_dim = dim // 2
    output = torch.empty_like(x)
    if num_tokens * num_heads == 0 or half_dim == 0:
        return output
    head_tiles = triton.cdiv(num_heads, _HEADS_TILE)
    total_tiles = num_tokens * head_tiles
    grid = (min(total_tiles, _MAX_GRID),)
    half_pow2 = triton.next_power_of_2(half_dim)
    _rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        output,
        total_tiles,
        head_tiles,
        num_heads,
        half_dim,
        HEADS_TILE=_HEADS_TILE,
        FULL=2 * half_pow2,
        HALF=half_pow2,
    )
    return output


__all__ = ["rotary_embedding"]

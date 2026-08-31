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
_HEADS_TILE_MAX = 16


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
    HALF_DIM: tl.constexpr,
):
    # enflame vendor (e4): the E2 generic [4, HALF_DIM] tile is only
    # ~2KB - far below the per-program workload enflame wants (T33 14x /
    # T39 BLOCK 4096 evidence); widen the head tile up to 16 (masked) so
    # each program owns up to [16, HALF_DIM] with one cos/sin load
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    h_offs = tl.arange(0, HEADS_TILE)
    i = tl.arange(0, HALF_DIM)
    imask = i < half_dim
    for tile in range(pid, total_tiles, grid_stride):
        t = tile // head_tiles
        head_tile = tile - t * head_tiles
        h0 = head_tile * HEADS_TILE
        hmask = h0 + h_offs < num_heads
        mask2d = hmask[:, None] & imask[None, :]
        rows = t * num_heads + h0 + h_offs
        x_base = x_ptr + rows[:, None] * (2 * half_dim) + (2 * i + 1)[None, :]
        x1 = tl.load(x_base - 1, mask=mask2d, other=0.0).to(tl.float32)
        x2 = tl.load(x_base, mask=mask2d, other=0.0).to(tl.float32)
        cs_base = t * half_dim + i
        c = tl.load(cos_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        o1 = x1 * c[None, :] - x2 * s[None, :]
        o2 = x1 * s[None, :] + x2 * c[None, :]
        out_base = (
            out_ptr + rows[:, None] * (2 * half_dim) + (2 * i + 1)[None, :]
        )
        out_ty = out_ptr.dtype.element_ty
        tl.store(out_base - 1, o1.to(out_ty), mask=mask2d)
        tl.store(out_base, o2.to(out_ty), mask=mask2d)


def rotary_embedding(x, cos, sin, interleaved):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    num_tokens, num_heads, dim = x.shape
    half_dim = dim // 2
    output = torch.empty_like(x)
    if num_tokens * num_heads == 0 or half_dim == 0:
        return output
    heads_tile = min(triton.next_power_of_2(num_heads), _HEADS_TILE_MAX)
    head_tiles = triton.cdiv(num_heads, heads_tile)
    total_tiles = num_tokens * head_tiles
    grid = (min(total_tiles, _MAX_GRID),)
    _rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        output,
        total_tiles,
        head_tiles,
        num_heads,
        half_dim,
        HEADS_TILE=heads_tile,
        HALF_DIM=triton.next_power_of_2(half_dim),
    )
    return output


__all__ = ["rotary_embedding"]

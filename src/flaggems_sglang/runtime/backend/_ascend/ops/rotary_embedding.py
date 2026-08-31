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
_HEADS_TILE = 4


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
    # ascend vendor (e3): keeps the E1 cos/sin reuse (loaded once per
    # HEADS_TILE heads - the mechanism that won +48~89% on six chips) but
    # drops the [HEADS_TILE, HALF_DIM] 2D broadcast tile that triggered the
    # ascend lowering NaN; every access stays 1D like the S0 kernel that
    # already passes on huawei, and static_range per-head amortization is
    # platform-neutral on ascend (T33 e1 evidence)
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    i = tl.arange(0, HALF_DIM)
    imask = i < half_dim
    out_ty = out_ptr.dtype.element_ty
    for tile in range(pid, total_tiles, grid_stride):
        t = tile // head_tiles
        head_tile = tile - t * head_tiles
        h0 = head_tile * HEADS_TILE
        cs_base = t * half_dim + i
        c = tl.load(cos_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_base, mask=imask, other=0.0).to(tl.float32)
        for hh in tl.static_range(HEADS_TILE):
            hmask = (h0 + hh < num_heads) & imask
            row_base = (t * num_heads + h0 + hh) * (2 * half_dim)
            x_base = x_ptr + row_base + 2 * i
            x1 = tl.load(x_base, mask=hmask, other=0.0).to(tl.float32)
            x2 = tl.load(x_base + 1, mask=hmask, other=0.0).to(tl.float32)
            o1 = x1 * c - x2 * s
            o2 = x1 * s + x2 * c
            out_base = out_ptr + row_base + 2 * i
            tl.store(out_base, o1.to(out_ty), mask=hmask)
            tl.store(out_base + 1, o2.to(out_ty), mask=hmask)


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
        HALF_DIM=triton.next_power_of_2(half_dim),
    )
    return output


__all__ = ["rotary_embedding"]

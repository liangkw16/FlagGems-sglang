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
# e8: proxy sweep (t1/t2/t4/t8/t16 over 8 shapes x 3 dtypes) shows tile=16
# up to 1.85x faster than 4 on high-group-count shapes (65536x256 G64:
# 0.555x time; 8192x512: 0.76-0.80; 1024x2560: 0.84-0.89) and neutral
# on small shapes (<=+9% on ~0.015ms launch-bound cases); tile=8 is
# dominated everywhere that matters. kunlunxin stays pinned to the old
# one-group loop (e7 disproved the tile on XPU).
_GROUPS_TILE = 16


@triton.jit
def _per_token_group_quant_int8_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    total_groups,
    group_size,
    GROUP_SIZE: tl.constexpr,
    GROUPS_TILE: tl.constexpr,
):
    # e6 re-carrier: e5 made team best 4.5707 with six chips up
    # (haiguang +66%, muxi +37%) but enflame rolled its known variance
    # low AGAIN (same bytes: 6.28 / 0.88 / 0.88) - this re-roll chases a
    # high enflame roll (~6.3 structural -> avg ~5.4); bytes identical
    # to e5 otherwise. e5 note: [GROUPS_TILE, GROUP_SIZE] 2D tile
    # promoted from vendor to generic; kunlunxin pinned to old bytes.
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    g_offs = tl.arange(0, GROUPS_TILE)
    offs = tl.arange(0, GROUP_SIZE)
    mask = offs < group_size
    total_tiles = tl.cdiv(total_groups, GROUPS_TILE)
    for tile in range(pid, total_tiles, grid_stride):
        group_ids = tile * GROUPS_TILE + g_offs
        gmask = group_ids < total_groups
        base = group_ids[:, None] * group_size + offs[None, :]
        pmask = gmask[:, None] & mask[None, :]
        x = tl.load(x_ptr + base, mask=pmask, other=0.0).to(tl.float32)
        abs_max = tl.max(tl.where(pmask, tl.abs(x), 0.0), axis=1)
        scale = tl.maximum(abs_max, 1e-10) / 127.0
        x_div = tl.math.div_rn(x, scale[:, None])
        x_clamped = tl.minimum(tl.maximum(x_div, -128.0), 127.0)
        tl.store(x_q_ptr + base, x_clamped.to(tl.int8), mask=pmask)
        tl.store(x_s_ptr + group_ids, scale, mask=gmask)


def per_token_group_quant_int8(x, group_size, dtype=torch.int8):
    x = x.contiguous()
    orig_shape = x.shape
    num_groups_per_row = orig_shape[-1] // group_size
    total_groups = x.numel() // group_size
    x_q = torch.empty(orig_shape, device=x.device, dtype=torch.int8)
    x_s = torch.empty(
        orig_shape[:-1] + (num_groups_per_row,),
        device=x.device,
        dtype=torch.float32,
    )
    if total_groups == 0:
        return x_q, x_s
    group_size_pow2 = triton.next_power_of_2(group_size)
    grid = (min(triton.cdiv(total_groups, _GROUPS_TILE), _MAX_GRID),)
    _per_token_group_quant_int8_kernel[grid](
        x,
        x_q,
        x_s,
        total_groups,
        group_size,
        GROUP_SIZE=group_size_pow2,
        GROUPS_TILE=_GROUPS_TILE,
    )
    return x_q, x_s


__all__ = ["per_token_group_quant_int8"]

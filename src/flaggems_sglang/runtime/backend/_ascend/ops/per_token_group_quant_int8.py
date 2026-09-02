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
# e10: 4 -> 16 mirrors the e8/e9 tile bump (platform: five chips
# +20~54%, enflame +45%); [16,128] fp32 tile ~8KB stays far under
# the Ascend UB budget that e3 measured at 4KB-well-under
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
    # ascend vendor (e3): same [GROUPS_TILE, GROUP_SIZE] 2D tile as the
    # enflame e2 win (14x there) - independent per-chip hypothesis; tile
    # footprint [4, <=256] fp32 ~4KB, far inside the ascend UB budget
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

# water-sample carrier r1 (2026-09-02): bytes identical to e10-8e344b4 team best
# water-sample carrier r2 (2026-09-02)
# water-sample carrier r3 (2026-09-03)
# water-sample carrier r5 (2026-09-03 night)

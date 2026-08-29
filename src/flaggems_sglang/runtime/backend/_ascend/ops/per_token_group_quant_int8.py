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
_GROUPS_PER_PROG = 8


@triton.jit
def _per_token_group_quant_int8_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    total_groups,
    group_size,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0) * GROUPS_PER_PROG
    offs = tl.arange(0, GROUP_SIZE)
    mask = offs < group_size
    for tile in range(pid, total_groups, grid_stride):
        for gg in tl.static_range(GROUPS_PER_PROG):
            group_id = tile * GROUPS_PER_PROG + gg
            gmask = mask & (group_id < total_groups)
            base = group_id * group_size
            x = tl.load(x_ptr + base + offs, mask=gmask, other=0.0).to(
                tl.float32
            )
            abs_max = tl.max(tl.where(gmask, tl.abs(x), 0.0), axis=0)
            scale = tl.maximum(abs_max, 1e-10) / 127.0
            x_div = tl.math.div_rn(x, scale)
            x_clamped = tl.minimum(tl.maximum(x_div, -128.0), 127.0)
            tl.store(x_q_ptr + base + offs, x_clamped.to(tl.int8), mask=gmask)
            tl.store(x_s_ptr + group_id, scale, mask=group_id < total_groups)


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
    grid = (min(triton.cdiv(total_groups, _GROUPS_PER_PROG), _MAX_GRID),)
    _per_token_group_quant_int8_kernel[grid](
        x,
        x_q,
        x_s,
        total_groups,
        group_size,
        GROUP_SIZE=group_size_pow2,
        GROUPS_PER_PROG=_GROUPS_PER_PROG,
    )
    return x_q, x_s


__all__ = ["per_token_group_quant_int8"]

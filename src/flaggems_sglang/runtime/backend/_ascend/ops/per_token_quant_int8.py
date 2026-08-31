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
_BLOCK = 512


@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    total_groups,
    group_size,
    COL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # ascend vendor (e1): two-pass column-blocked scan of each row at the
    # T24-proven ascend BLOCK 512 instead of one next_pow2(N)-wide tile;
    # chunk-wise running max is order-invariant, all math (fp32 amax,
    # div_rn, truncating int8 cast) is identical to the generic kernel
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for group_id in range(pid, total_groups, grid_stride):
        base = group_id * group_size
        abs_max = tl.zeros((), dtype=tl.float32)
        for cb in tl.static_range(COL_BLOCKS):
            offs_cb = cb * BLOCK + offs
            mask = offs_cb < group_size
            x = tl.load(x_ptr + base + offs_cb, mask=mask, other=0.0).to(
                tl.float32
            )
            abs_max = tl.maximum(abs_max, tl.max(tl.abs(x), axis=0))
        scale = tl.maximum(abs_max, 1e-10) / 127.0
        for cb in tl.static_range(COL_BLOCKS):
            offs_cb = cb * BLOCK + offs
            mask = offs_cb < group_size
            x = tl.load(x_ptr + base + offs_cb, mask=mask, other=0.0).to(
                tl.float32
            )
            x_div = tl.math.div_rn(x, scale)
            x_clamped = tl.minimum(tl.maximum(x_div, -128.0), 127.0)
            tl.store(
                x_q_ptr + base + offs_cb, x_clamped.to(tl.int8), mask=mask
            )
        tl.store(x_s_ptr + group_id, scale)


def per_token_quant_int8(x):
    x = x.contiguous()
    orig_shape = x.shape
    group_size = orig_shape[-1]
    total_groups = x.numel() // group_size
    x_q = torch.empty(orig_shape, device=x.device, dtype=torch.int8)
    x_s = torch.empty(
        orig_shape[:-1] + (1,), device=x.device, dtype=torch.float32
    )
    if total_groups == 0 or group_size == 0:
        return x_q, x_s
    grid = (min(total_groups, _MAX_GRID),)
    _per_token_quant_int8_kernel[grid](
        x,
        x_q,
        x_s,
        total_groups,
        group_size,
        COL_BLOCKS=triton.cdiv(group_size, _BLOCK),
        BLOCK=_BLOCK,
    )
    return x_q, x_s


__all__ = ["per_token_quant_int8"]

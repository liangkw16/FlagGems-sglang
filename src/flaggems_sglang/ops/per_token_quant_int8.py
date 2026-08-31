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
# e3 row-pack: single-pass row amax forces the whole row into one
# program, so the only single-pass generalization is packing ROWS rows
# per program. Host-side static dispatch (never per-element predicates
# - T32 e4 tuition). First screening attempt packed small row counts
# too and underfilled the device (1024-row shapes regressed 5-12% with
# only ~64 programs): packing pays only when total rows are huge, where
# fewer-but-fatter programs beat scheduling overhead (65536x256 ran
# 0.463x time, 2.16x faster, at ROWS=8). Row threshold 16384 keeps
# every other shape on the S0 kernel - byte-identical behavior.
_ROW_PACK_TARGET_ELEMS = 2048
_ROW_PACK_MAX_ROWS = 16
_ROW_PACK_MIN_ROWS = 16384


@triton.jit
def _per_token_quant_int8_rows_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    total_rows,
    row_len,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    # [ROWS, GROUP_SIZE] 2D tile, parallel axis-1 amax - the T33 e8
    # structure applied to whole rows; div_rn keeps torch bit-exactness
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    r_offs = tl.arange(0, ROWS)
    offs = tl.arange(0, GROUP_SIZE)
    cmask = offs < row_len
    total_tiles = tl.cdiv(total_rows, ROWS)
    for tile in range(pid, total_tiles, grid_stride):
        row_ids = tile * ROWS + r_offs
        rmask = row_ids < total_rows
        pmask = rmask[:, None] & cmask[None, :]
        base = row_ids[:, None] * row_len + offs[None, :]
        x = tl.load(x_ptr + base, mask=pmask, other=0.0).to(tl.float32)
        abs_max = tl.max(tl.where(pmask, tl.abs(x), 0.0), axis=1)
        scale = tl.maximum(abs_max, 1e-10) / 127.0
        x_div = tl.math.div_rn(x, scale[:, None])
        x_clamped = tl.minimum(tl.maximum(x_div, -128.0), 127.0)
        tl.store(x_q_ptr + base, x_clamped.to(tl.int8), mask=pmask)
        tl.store(x_s_ptr + row_ids, scale, mask=rmask)


@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    total_groups,
    group_size,
    GROUP_SIZE: tl.constexpr,
):
    # identical math to the platform-proven per_token_group_quant_int8
    # S0 kernel (whole-row groups): amax -> scale -> IEEE div_rn ->
    # clamp -> truncating int8 cast
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs = tl.arange(0, GROUP_SIZE)
    mask = offs < group_size
    for group_id in range(pid, total_groups, grid_stride):
        base = group_id * group_size
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        abs_max = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(abs_max, 1e-10) / 127.0
        x_div = tl.math.div_rn(x, scale)
        x_clamped = tl.minimum(tl.maximum(x_div, -128.0), 127.0)
        tl.store(x_q_ptr + base + offs, x_clamped.to(tl.int8), mask=mask)
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
    group_pow2 = triton.next_power_of_2(group_size)
    rows_tile = 1
    if total_groups >= _ROW_PACK_MIN_ROWS:
        rows_tile = min(
            _ROW_PACK_TARGET_ELEMS // group_pow2, _ROW_PACK_MAX_ROWS
        )
    if rows_tile > 1:
        grid = (min(triton.cdiv(total_groups, rows_tile), _MAX_GRID),)
        _per_token_quant_int8_rows_kernel[grid](
            x,
            x_q,
            x_s,
            total_groups,
            group_size,
            GROUP_SIZE=group_pow2,
            ROWS=rows_tile,
        )
        return x_q, x_s
    grid = (min(total_groups, _MAX_GRID),)
    _per_token_quant_int8_kernel[grid](
        x,
        x_q,
        x_s,
        total_groups,
        group_size,
        GROUP_SIZE=group_pow2,
    )
    return x_q, x_s


__all__ = ["per_token_quant_int8"]

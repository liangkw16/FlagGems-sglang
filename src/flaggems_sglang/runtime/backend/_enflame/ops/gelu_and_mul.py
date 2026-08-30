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

# enflame vendor (e8): BLOCK_COL 4096 covers a whole row in one
# program - the T39 e8 platform-proven enflame optimum (+90% there)
_BLOCK_COL = 4096
_MAX_GRID = 65535


@triton.jit
def _gelu_and_mul_kernel(
    x_ptr,
    output_ptr,
    rows,
    half_width,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(half_width, BLOCK_COL)
    total_blocks = rows * num_col_blocks
    grid_size = tl.num_programs(0)
    for block_id in range(pid, total_blocks, grid_size):
        row_id = block_id // num_col_blocks
        col_block = block_id - row_id * num_col_blocks
        col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
        col_mask = col_offsets < half_width
        row_base = row_id.to(tl.int64) * (2 * half_width)
        gate = tl.load(
            x_ptr + row_base + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            x_ptr + row_base + half_width + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        # Abramowitz-Stegun 7.1.26 rational erf approximation,
        # max abs error < 1.5e-7 (fp32 tolerance 1e-4); avoids
        # tl.math.erf which crashes the kunlunxin compiler.
        scaled = gate * 0.7071067811865476
        abs_scaled = tl.abs(scaled)
        t = 1.0 / (1.0 + 0.3275911 * abs_scaled)
        poly = (
            (
                ((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                - 0.284496736
            )
            * t
            + 0.254829592
        ) * t
        erf_abs = 1.0 - poly * tl.exp(-abs_scaled * abs_scaled)
        erf_scaled = tl.where(scaled < 0.0, -erf_abs, erf_abs)
        gelu = gate * 0.5 * (1.0 + erf_scaled)
        tl.store(
            output_ptr + row_id.to(tl.int64) * half_width + col_offsets,
            (gelu * up).to(output_ptr.dtype.element_ty),
            mask=col_mask,
        )


def gelu_and_mul(hidden_states):
    x = hidden_states.contiguous()
    last_dim = x.shape[-1]
    half_width = last_dim // 2
    output = torch.empty(
        x.shape[:-1] + (half_width,), dtype=x.dtype, device=x.device
    )
    if last_dim == 0:
        return output
    rows = x.numel() // last_dim
    if rows * half_width == 0:
        return output
    num_col_blocks = triton.cdiv(half_width, _BLOCK_COL)
    total_blocks = rows * num_col_blocks
    grid = (min(total_blocks, _MAX_GRID),)
    _gelu_and_mul_kernel[grid](
        x,
        output,
        rows,
        half_width,
        BLOCK_COL=_BLOCK_COL,
    )
    return output


__all__ = ["gelu_and_mul"]

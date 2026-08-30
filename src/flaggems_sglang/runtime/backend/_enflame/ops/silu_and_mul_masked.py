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

# enflame vendor (e8): BLOCK_COL 4096 - T24 platform-proven enflame
# optimum for double-read elementwise; structure otherwise identical
_BLOCK_COL = 4096
_MAX_GRID = 65535


@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    masked_m_ptr,
    output_ptr,
    total_blocks,
    tokens,
    half_width,
    num_col_blocks,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_COL)
    for block_id in range(pid, total_blocks, grid_stride):
        row_id = block_id // num_col_blocks
        col_block = block_id - row_id * num_col_blocks
        expert_id = row_id // tokens
        token_id = row_id - expert_id * tokens
        cols = col_block * BLOCK_COL + offsets
        valid_rows = tl.load(masked_m_ptr + expert_id)
        mask = (token_id < valid_rows) & (cols < half_width)
        input_base = row_id.to(tl.int64) * (2 * half_width)
        gate = tl.load(input_ptr + input_base + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            input_ptr + input_base + half_width + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = (gate / (1.0 + tl.exp(-gate))) * up
        tl.store(
            output_ptr + row_id.to(tl.int64) * half_width + cols,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def silu_and_mul_masked(input, masked_m):
    input = input.contiguous()
    masked_m = masked_m.contiguous()
    experts, tokens, width = input.shape
    half_width = width // 2
    output = torch.empty(
        (experts, tokens, half_width),
        dtype=torch.bfloat16,
        device=input.device,
    )
    if experts == 0 or tokens == 0 or half_width == 0:
        return output

    num_col_blocks = triton.cdiv(half_width, _BLOCK_COL)
    total_blocks = experts * tokens * num_col_blocks
    _silu_and_mul_masked_kernel[(min(total_blocks, _MAX_GRID),)](
        input,
        masked_m,
        output,
        total_blocks,
        tokens,
        half_width,
        num_col_blocks,
        BLOCK_COL=_BLOCK_COL,
    )
    return output


__all__ = ["silu_and_mul_masked"]

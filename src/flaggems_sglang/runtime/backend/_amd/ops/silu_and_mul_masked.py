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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_COL": 128}, num_warps=2),
        triton.Config({"BLOCK_COL": 256}, num_warps=4),
        triton.Config({"BLOCK_COL": 512}, num_warps=8),
        triton.Config({"BLOCK_COL": 1024}, num_warps=8),
    ],
    key=["half_width"],
)
@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    masked_m_ptr,
    output_ptr,
    num_rows,
    tokens,
    half_width,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    num_col_blocks = tl.cdiv(half_width, BLOCK_COL)
    total_blocks = num_rows * num_col_blocks
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

    num_rows = experts * tokens
    grid = lambda meta: (  # noqa: E731
        min(
            num_rows * triton.cdiv(half_width, meta["BLOCK_COL"]),
            _MAX_GRID,
        ),
    )
    _silu_and_mul_masked_kernel[grid](
        input,
        masked_m,
        output,
        num_rows,
        tokens,
        half_width,
    )
    return output


__all__ = ["silu_and_mul_masked"]
# anchor-hunt carrier (2026-09-03): bytes identical to e10-b9a48a2
# anchor-hunt final (2026-09-03)
# anchor-hunt final 2 (2026-09-03)

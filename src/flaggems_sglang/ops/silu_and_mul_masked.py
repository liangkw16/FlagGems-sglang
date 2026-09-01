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

_BLOCK_COL = 1024
_MAX_GRID = 65535
# e10 token-block skip: each program owns a [ROWS_PER_BLOCK, BLOCK_COL]
# slab. A block whose first row is already past masked_m[e] exits after
# one scalar load (padding costs a launch, not an iteration), and the
# CTA count drops by ROWS_PER_BLOCK versus the E7 one-row-per-block
# flat kernel - heavy-padding shapes were launch-throughput bound. The
# per-row guard is a uniform scalar branch (not a per-element load
# predicate - T32 e4 tuition). A sync-free design: the first screening
# attempt used a prefix-sum + binary search + .item() and the ~25us
# host sync regressed every moderate-padding case by 16-41%.
_ROWS_PER_BLOCK = 8


@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    masked_m_ptr,
    output_ptr,
    total_blocks,
    tokens,
    half_width,
    num_row_blocks,
    num_col_blocks,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_COL)
    for block_id in range(pid, total_blocks, grid_stride):
        rc = block_id // num_col_blocks
        col_block = block_id - rc * num_col_blocks
        expert_id = rc // num_row_blocks
        row_block = rc - expert_id * num_row_blocks
        valid_rows = tl.load(masked_m_ptr + expert_id)
        row_lo = row_block * ROWS_PER_BLOCK
        if row_lo < valid_rows:
            cols = col_block * BLOCK_COL + offsets
            cmask = cols < half_width
            for r in tl.static_range(ROWS_PER_BLOCK):
                token_id = row_lo + r
                if token_id < valid_rows:
                    row_id = expert_id * tokens + token_id
                    input_base = row_id.to(tl.int64) * (2 * half_width)
                    gate = tl.load(
                        input_ptr + input_base + cols,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    up = tl.load(
                        input_ptr + input_base + half_width + cols,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    output = (gate / (1.0 + tl.exp(-gate))) * up
                    tl.store(
                        output_ptr + row_id.to(tl.int64) * half_width + cols,
                        output.to(output_ptr.dtype.element_ty),
                        mask=cmask,
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

    num_row_blocks = triton.cdiv(tokens, _ROWS_PER_BLOCK)
    num_col_blocks = triton.cdiv(half_width, _BLOCK_COL)
    total_blocks = experts * num_row_blocks * num_col_blocks
    _silu_and_mul_masked_kernel[(min(total_blocks, _MAX_GRID),)](
        input,
        masked_m,
        output,
        total_blocks,
        tokens,
        half_width,
        num_row_blocks,
        num_col_blocks,
        ROWS_PER_BLOCK=_ROWS_PER_BLOCK,
        BLOCK_COL=_BLOCK_COL,
    )
    return output


__all__ = ["silu_and_mul_masked"]

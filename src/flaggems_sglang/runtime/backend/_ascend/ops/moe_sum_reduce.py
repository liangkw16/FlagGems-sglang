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


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    hidden_dim,
    hidden_blocks,
    total_programs,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_stride_token = tl.cast(input_stride_token, tl.int64)
    input_stride_topk = tl.cast(input_stride_topk, tl.int64)
    input_stride_hidden = tl.cast(input_stride_hidden, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_hidden = tl.cast(output_stride_hidden, tl.int64)

    program = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for logical_id in range(program, total_programs, grid_size):
        token_offset = logical_id // hidden_blocks
        hidden_block = logical_id % hidden_blocks
        hidden_offsets = hidden_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hidden_mask = hidden_offsets < hidden_dim
        input_offsets = (
            token_offset * input_stride_token
            + hidden_offsets * input_stride_hidden
        )
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for expert_offset in range(TOP_K):
            values = tl.load(
                input_ptr + input_offsets + expert_offset * input_stride_topk,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += values

        output_offsets = (
            token_offset * output_stride_token
            + hidden_offsets * output_stride_hidden
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator * ROUTED_SCALING_FACTOR,
            mask=hidden_mask,
        )


def moe_sum_reduce(input, routed_scaling_factor):
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    if num_tokens == 0 or hidden_dim == 0:
        return output

    block_size = 512
    hidden_blocks = triton.cdiv(hidden_dim, block_size)
    total_programs = num_tokens * hidden_blocks
    grid = (min(total_programs, 4096),)
    _moe_sum_reduce_kernel[grid](
        input,
        output,
        *input.stride(),
        *output.stride(),
        hidden_dim,
        hidden_blocks,
        total_programs,
        ROUTED_SCALING_FACTOR=routed_scaling_factor,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["moe_sum_reduce"]

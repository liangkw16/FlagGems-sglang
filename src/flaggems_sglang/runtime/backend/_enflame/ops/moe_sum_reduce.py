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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "topk"],
)
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    hidden_size,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_stride_token = tl.cast(input_stride_token, tl.int64)
    input_stride_topk = tl.cast(input_stride_topk, tl.int64)
    input_stride_hidden = tl.cast(input_stride_hidden, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_hidden = tl.cast(output_stride_hidden, tl.int64)

    token_offset = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    input_offsets = (
        token_offset * input_stride_token
        + hidden_offsets * input_stride_hidden
    )
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for expert_offset in range(topk):
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
    num_tokens, top_k, hidden_size = input.shape
    output = torch.empty(
        (num_tokens, hidden_size), dtype=input.dtype, device=input.device
    )
    if num_tokens == 0 or hidden_size == 0:
        return output

    grid = lambda meta: (
        num_tokens,
        triton.cdiv(hidden_size, meta["BLOCK_SIZE"]),
    )
    _moe_sum_reduce_kernel[grid](
        input,
        output,
        *input.stride(),
        *output.stride(),
        hidden_size,
        ROUTED_SCALING_FACTOR=routed_scaling_factor,
        topk=top_k,
    )
    return output


__all__ = ["moe_sum_reduce"]

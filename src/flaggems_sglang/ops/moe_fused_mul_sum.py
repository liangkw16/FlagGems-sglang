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
def _moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    output_ptr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    weight_stride_token,
    weight_stride_topk,
    ids_stride_token,
    ids_stride_topk,
    hidden_dim,
    scale,
    HAS_EXPERT_MAP: tl.constexpr,
    IS_EP: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    input_stride_token = tl.cast(input_stride_token, tl.int64)
    input_stride_topk = tl.cast(input_stride_topk, tl.int64)
    input_stride_hidden = tl.cast(input_stride_hidden, tl.int64)

    token_offset = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_dim
    input_row = (
        token_offset * input_stride_token
        + hidden_offsets * input_stride_hidden
    )

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for expert_offset in range(TOP_K):
        weight = (
            tl.load(
                topk_weights_ptr
                + token_offset * weight_stride_token
                + expert_offset * weight_stride_topk
            ).to(tl.float32)
            * scale
        )

        if HAS_EXPERT_MAP:
            expert_id = tl.load(
                topk_ids_ptr
                + token_offset * ids_stride_token
                + expert_offset * ids_stride_topk
            )
            mapped = tl.load(expert_map_ptr + expert_id)
            weight = tl.where(mapped >= 0, weight, 0.0)
        elif IS_EP:
            expert_id = tl.load(
                topk_ids_ptr
                + token_offset * ids_stride_token
                + expert_offset * ids_stride_topk
            )
            weight = tl.where(expert_id >= 0, weight, 0.0)

        values = tl.load(
            inputs_ptr + input_row + expert_offset * input_stride_topk,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values * weight

    output_offsets = token_offset.to(tl.int64) * hidden_dim + hidden_offsets
    tl.store(output_ptr + output_offsets, accumulator, mask=hidden_mask)


def moe_fused_mul_sum(
    inputs,
    topk_weights,
    topk_ids=None,
    expert_map=None,
    routed_scaling_factor=None,
    is_ep=False,
):
    num_tokens, top_k, hidden_dim = inputs.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=inputs.dtype, device=inputs.device
    )
    if num_tokens == 0 or hidden_dim == 0:
        return output

    scale = (
        1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    )
    has_expert_map = expert_map is not None

    if topk_ids is not None:
        topk_ids_arg = topk_ids
        ids_stride_token, ids_stride_topk = topk_ids.stride(
            0
        ), topk_ids.stride(1)
    else:
        topk_ids_arg = torch.empty(0, dtype=torch.int32, device=inputs.device)
        ids_stride_token, ids_stride_topk = 0, 0
    if not has_expert_map:
        expert_map = torch.empty(0, dtype=torch.int32, device=inputs.device)

    block_size = 256
    grid = (num_tokens, triton.cdiv(hidden_dim, block_size))
    _moe_fused_mul_sum_kernel[grid](
        inputs,
        topk_weights,
        topk_ids_arg,
        expert_map,
        output,
        *inputs.stride(),
        *topk_weights.stride(),
        ids_stride_token,
        ids_stride_topk,
        hidden_dim,
        scale,
        HAS_EXPERT_MAP=has_expert_map,
        IS_EP=is_ep and topk_ids is not None and not has_expert_map,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["moe_fused_mul_sum"]

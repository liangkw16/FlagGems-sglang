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
def _decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    output_ptr,
    query_heads,
    kv_heads,
    qk_dim,
    value_dim,
    sm_scale,
    q_stride_batch,
    q_stride_head,
    q_stride_dim,
    k_stride_page,
    k_stride_head,
    k_stride_dim,
    v_stride_page,
    v_stride_head,
    v_stride_dim,
    indptr_stride,
    indices_stride,
    output_stride_batch,
    output_stride_head,
    output_stride_dim,
    BLOCK_LENGTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch = program_id // query_heads
    query_head = program_id % query_heads
    kv_head = query_head // (query_heads // kv_heads)

    start = tl.load(indptr_ptr + batch * indptr_stride).to(tl.int32)
    end = tl.load(indptr_ptr + (batch + 1) * indptr_stride).to(tl.int32)
    sequence_length = end - start

    dim = tl.arange(0, BLOCK_D)
    dim_mask = dim < qk_dim
    query = tl.load(
        q_ptr
        + batch * q_stride_batch
        + query_head * q_stride_head
        + dim * q_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    value_offset = tl.arange(0, BLOCK_DV)
    value_mask = value_offset < value_dim
    maximum = float("-inf")
    denominator = 0.0
    accumulator = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for block_start in range(0, sequence_length, BLOCK_LENGTH):
        positions = block_start + tl.arange(0, BLOCK_LENGTH)
        position_mask = positions < sequence_length
        pages = tl.load(
            indices_ptr + (start + positions) * indices_stride,
            mask=position_mask,
            other=0,
        ).to(tl.int32)

        keys = tl.load(
            k_ptr
            + pages[:, None] * k_stride_page
            + kv_head * k_stride_head
            + dim[None, :] * k_stride_dim,
            mask=position_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(query[None, :] * keys, axis=1) * sm_scale
        logits = tl.where(position_mask, logits, float("-inf"))

        new_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
        correction = tl.exp(maximum - new_maximum)
        probabilities = tl.exp(logits - new_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)

        values = tl.load(
            v_ptr
            + pages[:, None] * v_stride_page
            + kv_head * v_stride_head
            + value_offset[None, :] * v_stride_dim,
            mask=position_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        maximum = new_maximum

    output = accumulator / denominator
    tl.store(
        output_ptr
        + batch * output_stride_batch
        + query_head * output_stride_head
        + value_offset * output_stride_dim,
        output,
        mask=value_mask,
    )


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, qk_dim = q.shape
    kv_heads = k_buffer.shape[1]
    value_dim = v_buffer.shape[-1]
    output = torch.empty(
        (batch_size, query_heads, value_dim),
        dtype=torch.float32,
        device=q.device,
    )
    if output.numel() == 0:
        return output

    routed_indptr = (
        kv_indptr.to(torch.int32)
        if kv_indptr.dtype == torch.int64
        else kv_indptr
    )
    routed_indices = (
        kv_indices.to(torch.int32)
        if kv_indices.dtype == torch.int64
        else kv_indices
    )
    block_d = triton.next_power_of_2(qk_dim)
    block_dv = triton.next_power_of_2(value_dim)
    block_length = max(8, min(32, 8192 // max(block_d, block_dv)))
    _decode_attention_kernel[(batch_size * query_heads,)](
        q,
        k_buffer,
        v_buffer,
        routed_indptr,
        routed_indices,
        output,
        query_heads,
        kv_heads,
        qk_dim,
        value_dim,
        float(sm_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        routed_indptr.stride(0),
        routed_indices.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_LENGTH=block_length,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["decode_attention"]

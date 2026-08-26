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

_BLOCK_DV = 64
_MAX_GRID_PROGRAMS = 65535


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
    program_start,
    value_tiles,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    program_id = program_start + tl.program_id(0)
    value_tile = program_id % value_tiles
    head_program = program_id // value_tiles
    batch = head_program // query_heads
    query_head = head_program % query_heads
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

    value_offset = value_tile * BLOCK_DV + tl.arange(0, BLOCK_DV)
    value_mask = value_offset < value_dim
    maximum = float("-inf")
    denominator = 0.0
    accumulator = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for position in range(0, sequence_length):
        page = tl.load(indices_ptr + (start + position) * indices_stride).to(
            tl.int32
        )
        key = tl.load(
            k_ptr
            + page * k_stride_page
            + kv_head * k_stride_head
            + dim * k_stride_dim,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(query * key, axis=0) * sm_scale
        new_maximum = tl.maximum(maximum, score)
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(score - new_maximum)
        denominator = denominator * correction + probability

        value = tl.load(
            v_ptr
            + page * v_stride_page
            + kv_head * v_stride_head
            + value_offset * v_stride_dim,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + probability * value
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
    value_tiles = triton.cdiv(value_dim, _BLOCK_DV)
    total_programs = batch_size * query_heads * value_tiles
    for program_start in range(0, total_programs, _MAX_GRID_PROGRAMS):
        program_count = min(_MAX_GRID_PROGRAMS, total_programs - program_start)
        _decode_attention_kernel[(program_count,)](
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
            program_start,
            value_tiles,
            BLOCK_D=block_d,
            BLOCK_DV=_BLOCK_DV,
            num_warps=2,
            num_stages=1,
        )
    return output


__all__ = ["decode_attention"]

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

_MAX_GRID_PROGRAMS = 65535


@triton.jit
def _context_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    start_ptr,
    length_ptr,
    output_ptr,
    sm_scale,
    batch_head_start,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    k_stride_token,
    k_stride_head,
    k_stride_dim,
    v_stride_token,
    v_stride_head,
    v_stride_dim,
    start_stride,
    length_stride,
    output_stride_token,
    output_stride_head,
    output_stride_dim,
    QUERY_SLOTS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_head_offset = program_id // QUERY_SLOTS
    query_slot = program_id - batch_head_offset * QUERY_SLOTS
    batch_head = batch_head_start + batch_head_offset
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    sequence_start = tl.load(start_ptr + batch * start_stride).to(tl.int32)
    sequence_length = tl.load(length_ptr + batch * length_stride).to(tl.int32)
    kv_head = query_head // GROUP_SIZE
    query_block = query_slot

    while query_block * BLOCK_M < sequence_length:
        query_position = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offset = tl.arange(0, BLOCK_N)
        dim = tl.arange(0, BLOCK_D)
        query_mask = query_position < sequence_length
        dim_mask = dim < HEAD_DIM

        query = tl.load(
            q_ptr
            + (sequence_start + query_position[:, None]) * q_stride_token
            + query_head * q_stride_head
            + dim[None, :] * q_stride_dim,
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        negative_infinity = float("-inf")
        log2e = 1.4426950408889634
        running_maximum = tl.full(
            [BLOCK_M], negative_infinity, dtype=tl.float32
        )
        running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        if IS_CAUSAL:
            key_end = tl.minimum((query_block + 1) * BLOCK_M, sequence_length)
        else:
            key_end = sequence_length

        for key_start in range(0, key_end, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            key_position = key_start + key_offset
            key_mask = key_position < sequence_length

            keys = tl.load(
                k_ptr
                + (sequence_start + key_position[None, :]) * k_stride_token
                + kv_head * k_stride_head
                + dim[:, None] * k_stride_dim,
                mask=dim_mask[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(query, keys, input_precision="ieee") * (
                sm_scale * log2e
            )
            score_mask = query_mask[:, None] & key_mask[None, :]
            if IS_CAUSAL:
                score_mask &= query_position[:, None] >= key_position[None, :]
            scores = tl.where(score_mask, scores, negative_infinity)

            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(running_maximum, block_maximum)
            correction = tl.exp2(running_maximum - new_maximum)
            probabilities = tl.exp2(scores - new_maximum[:, None])

            values = tl.load(
                v_ptr
                + (sequence_start + key_position[:, None]) * v_stride_token
                + kv_head * v_stride_head
                + dim[None, :] * v_stride_dim,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction[:, None] + tl.dot(
                probabilities, values, input_precision="ieee"
            )
            running_sum = running_sum * correction + tl.sum(
                probabilities, axis=1
            )
            running_maximum = new_maximum

        result = accumulator / running_sum[:, None]
        tl.store(
            output_ptr
            + (sequence_start + query_position[:, None]) * output_stride_token
            + query_head * output_stride_head
            + dim[None, :] * output_stride_dim,
            result,
            mask=query_mask[:, None] & dim_mask[None, :],
        )
        query_block += QUERY_SLOTS


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 packed tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same token count")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q, k, and v must have the same head dimension")
    if b_start_loc.ndim != 1 or b_seq_len.ndim != 1:
        raise ValueError("b_start_loc and b_seq_len must be rank-1 tensors")
    if b_start_loc.numel() != b_seq_len.numel():
        raise ValueError("b_start_loc and b_seq_len must have equal length")

    total_tokens, query_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    if kv_heads == 0 or query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    output = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    batch_size = b_seq_len.numel()
    if output.numel() == 0 or batch_size == 0:
        return output

    block_m = 16
    block_n = 16
    block_d = max(16, triton.next_power_of_2(head_dim))
    planning_length = max(int(max_input_len), 1)
    query_slots = min(
        max(triton.cdiv(planning_length, block_m), 1),
        _MAX_GRID_PROGRAMS,
    )
    batch_heads = batch_size * query_heads
    batch_heads_per_launch = max(1, _MAX_GRID_PROGRAMS // query_slots)

    for batch_head_start in range(0, batch_heads, batch_heads_per_launch):
        batch_head_count = min(
            batch_heads_per_launch, batch_heads - batch_head_start
        )
        grid = (query_slots * batch_head_count,)
        _context_attention_kernel[grid](
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            output,
            head_dim**-0.5,
            batch_head_start,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            b_start_loc.stride(0),
            b_seq_len.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            QUERY_SLOTS=query_slots,
            QUERY_HEADS=query_heads,
            GROUP_SIZE=query_heads // kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            IS_CAUSAL=bool(is_causal),
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["context_attention"]

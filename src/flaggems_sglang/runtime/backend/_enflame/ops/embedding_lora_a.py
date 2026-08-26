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

_MAX_GRID_SIZE = 65535


@triton.jit
def _expand_routes_kernel(
    seg_indptr,
    weight_indices,
    lora_ranks,
    token_weight_indices,
    token_ranks,
    tiles_per_batch,
    program_start,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
    BLOCK_TOKENS: tl.constexpr,
):
    logical_id = program_start + tl.program_id(0)
    batch_id = logical_id // tiles_per_batch
    token_tile = logical_id - batch_id * tiles_per_batch
    segment_start = tl.load(seg_indptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    weight_index = tl.load(weight_indices + batch_id * weight_indices_stride)
    safe_weight_index = tl.where(segment_length > 0, weight_index, 0)
    rank = tl.load(lora_ranks + safe_weight_index * lora_ranks_stride)

    token_offsets = token_tile * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_offsets < segment_length
    tokens = segment_start + token_offsets
    tl.store(
        token_weight_indices + tokens,
        safe_weight_index.to(tl.int32),
        mask=token_mask,
    )
    tl.store(token_ranks + tokens, rank.to(tl.int32), mask=token_mask)


@triton.jit
def _gather_routes_kernel(
    input_ids,
    weights,
    output,
    extra_embeddings,
    token_weight_indices,
    token_ranks,
    vocab_size,
    rank_tiles,
    program_start,
    input_stride,
    weight_stride_lora,
    weight_stride_rank,
    weight_stride_vocab,
    output_stride_token,
    output_stride_rank,
    extra_stride_lora,
    extra_stride_token,
    extra_stride_rank,
    BLOCK_RANK: tl.constexpr,
    HAS_EXTRA_EMBEDDINGS: tl.constexpr,
):
    input_stride = tl.cast(input_stride, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_rank = tl.cast(weight_stride_rank, tl.int64)
    weight_stride_vocab = tl.cast(weight_stride_vocab, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_rank = tl.cast(output_stride_rank, tl.int64)
    extra_stride_lora = tl.cast(extra_stride_lora, tl.int64)
    extra_stride_token = tl.cast(extra_stride_token, tl.int64)
    extra_stride_rank = tl.cast(extra_stride_rank, tl.int64)

    logical_id = program_start + tl.program_id(0)
    token_offset = logical_id // rank_tiles
    rank_tile = logical_id - token_offset * rank_tiles
    rank_offsets = rank_tile * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    token_id = tl.load(input_ids + token_offset * input_stride)
    weight_index = tl.load(token_weight_indices + token_offset)
    rank = tl.load(token_ranks + token_offset)
    output_mask = rank_offsets < rank

    regular_id = tl.minimum(token_id, vocab_size - 1)
    if HAS_EXTRA_EMBEDDINGS:
        is_extra = token_id >= vocab_size
        regular = tl.load(
            weights
            + weight_index * weight_stride_lora
            + rank_offsets * weight_stride_rank
            + regular_id * weight_stride_vocab,
            mask=output_mask & (token_id < vocab_size),
            other=0.0,
        )
        extra_id = tl.maximum(token_id - vocab_size, 0)
        extra = tl.load(
            extra_embeddings
            + weight_index * extra_stride_lora
            + extra_id * extra_stride_token
            + rank_offsets * extra_stride_rank,
            mask=output_mask & is_extra,
            other=0.0,
        )
        values = tl.where(is_extra, extra, regular)
    else:
        values = tl.load(
            weights
            + weight_index * weight_stride_lora
            + rank_offsets * weight_stride_rank
            + regular_id * weight_stride_vocab,
            mask=output_mask,
            other=0.0,
        )

    output_offsets = (
        token_offset * output_stride_token + rank_offsets * output_stride_rank
    )
    tl.store(output + output_offsets, values, mask=output_mask)


def embedding_lora_a(
    input_ids, weights, batch_info, vocab_size, extra_embeddings=None
):
    output = torch.zeros(
        (input_ids.shape[0], weights.shape[1]),
        dtype=weights.dtype,
        device=weights.device,
    )
    if (
        input_ids.shape[0] == 0
        or batch_info.bs == 0
        or batch_info.max_len == 0
    ):
        return output

    route_block = 128
    tiles_per_batch = triton.cdiv(batch_info.max_len, route_block)
    route_programs = batch_info.bs * tiles_per_batch
    token_weight_indices = torch.zeros(
        input_ids.shape,
        dtype=torch.int32,
        device=input_ids.device,
    )
    token_ranks = torch.zeros(
        input_ids.shape,
        dtype=torch.int32,
        device=input_ids.device,
    )
    for program_start in range(0, route_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, route_programs - program_start),)
        _expand_routes_kernel[grid](
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            token_weight_indices,
            token_ranks,
            tiles_per_batch,
            program_start,
            batch_info.seg_indptr.stride(0),
            batch_info.weight_indices.stride(0),
            batch_info.lora_ranks.stride(0),
            BLOCK_TOKENS=route_block,
            num_warps=4,
            num_stages=1,
        )

    if extra_embeddings is None:
        extra_embeddings = output
        extra_strides = (0, 0, 0)
    else:
        extra_strides = extra_embeddings.stride()

    gather_block_rank = 128
    rank_tiles = triton.cdiv(weights.shape[1], gather_block_rank)
    gather_programs = input_ids.shape[0] * rank_tiles
    for program_start in range(0, gather_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, gather_programs - program_start),)
        _gather_routes_kernel[grid](
            input_ids,
            weights,
            output,
            extra_embeddings,
            token_weight_indices,
            token_ranks,
            vocab_size,
            rank_tiles,
            program_start,
            input_ids.stride(0),
            *weights.stride(),
            *output.stride(),
            *extra_strides,
            BLOCK_RANK=gather_block_rank,
            HAS_EXTRA_EMBEDDINGS=extra_embeddings is not output,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["embedding_lora_a"]

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


@triton.jit
def _chunked_embedding_lora_a_kernel(
    input_ids,
    weights,
    output,
    permutation,
    seg_indptr,
    weight_indices,
    lora_ranks,
    total_tokens,
    num_segments,
    ids_stride,
    perm_stride,
    seg_stride,
    widx_stride,
    ranks_stride,
    weight_stride_lora,
    weight_stride_rank,
    weight_stride_vocab,
    output_stride_token,
    output_stride_rank,
    LOG2_SEGMENTS: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_rank = tl.cast(weight_stride_rank, tl.int64)
    weight_stride_vocab = tl.cast(weight_stride_vocab, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_rank = tl.cast(output_stride_rank, tl.int64)

    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for pos in range(pid, total_tokens, grid_size):
        # Branchless binary search for the segment that owns pos:
        # largest b with seg_indptr[b] <= pos. Only unmasked scalar
        # loads; empty segments can never be selected, so their
        # (possibly sentinel) adapter metadata is never touched.
        b = 0
        span = 1
        for _ in tl.static_range(LOG2_SEGMENTS):
            span = span * 2
        for _ in tl.static_range(LOG2_SEGMENTS):
            span = span // 2
            mid = b + span
            clamped = tl.minimum(mid, num_segments)
            v = tl.load(seg_indptr + clamped * seg_stride)
            b = tl.where((mid <= num_segments) & (v <= pos), mid, b)

        weight_index = tl.load(weight_indices + b * widx_stride).to(tl.int64)
        rank = tl.load(lora_ranks + weight_index * ranks_stride)
        row = tl.load(permutation + pos * perm_stride).to(tl.int64)
        token_id = tl.load(input_ids + row * ids_stride).to(tl.int64)

        num_rank_blocks = tl.cdiv(rank, BLOCK_RANK)
        for rank_block in range(0, num_rank_blocks):
            rank_offsets = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
            rank_mask = rank_offsets < rank
            values = tl.load(
                weights
                + weight_index * weight_stride_lora
                + rank_offsets * weight_stride_rank
                + token_id * weight_stride_vocab,
                mask=rank_mask,
                other=0.0,
            )
            tl.store(
                output + row * output_stride_token + rank_offsets * output_stride_rank,
                values,
                mask=rank_mask,
            )


def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    total_tokens = input_ids.shape[0]
    max_rank = weights.shape[1]
    output = torch.zeros(
        (total_tokens, max_rank), dtype=weights.dtype, device=weights.device
    )
    num_segments = batch_info.bs
    if total_tokens == 0 or num_segments == 0:
        return output
    seg_pow2 = triton.next_power_of_2(num_segments + 1)
    log2_segments = max(seg_pow2.bit_length() - 1, 1)
    grid = (min(total_tokens, _MAX_GRID),)
    _chunked_embedding_lora_a_kernel[grid](
        input_ids,
        weights,
        output,
        batch_info.permutation,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        total_tokens,
        num_segments,
        input_ids.stride(0),
        batch_info.permutation.stride(0),
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        *weights.stride(),
        *output.stride(),
        LOG2_SEGMENTS=log2_segments,
        BLOCK_RANK=128,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

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

# Kunlunxin vendor: segment-owned persistent workers (12 workers matches the FlagTree XPU worker count) -
# each worker claims whole segments, reads segment bounds and adapter
# metadata once, then walks the segment's tokens; no per-token binary
# search and no dispatch cost for tens of thousands of logical
# programs.

import torch
import triton
import triton.language as tl

_WORKERS = 12


@triton.jit
def _cela_segment_owned_kernel(
    input_ids,
    weights,
    output,
    permutation,
    seg_indptr,
    weight_indices,
    lora_ranks,
    num_segments,
    num_lora,
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
    BLOCK_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)
    for seg in range(pid, num_segments, num_workers):
        start = tl.load(seg_indptr + seg * seg_stride)
        end = tl.load(seg_indptr + (seg + 1) * seg_stride)
        w_idx = tl.load(weight_indices + seg * widx_stride)
        # empty segments may carry sentinel adapter indices; clamp so
        # the ranks gather stays in bounds (their token loop is empty)
        w_idx = tl.minimum(w_idx, num_lora - 1)
        rank = tl.load(lora_ranks + w_idx * ranks_stride)
        w_idx64 = w_idx.to(tl.int64)
        num_rank_blocks = tl.cdiv(rank, BLOCK_RANK)
        for local in range(0, end - start):
            row = tl.load(permutation + (start + local) * perm_stride).to(tl.int64)
            token_id = tl.load(input_ids + row * ids_stride).to(tl.int64)
            for rank_block in range(0, num_rank_blocks):
                rank_offsets = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
                rank_mask = rank_offsets < rank
                values = tl.load(
                    weights
                    + w_idx64 * weight_stride_lora
                    + rank_offsets * weight_stride_rank
                    + token_id * weight_stride_vocab,
                    mask=rank_mask,
                    other=0.0,
                )
                tl.store(
                    output
                    + row * output_stride_token
                    + rank_offsets * output_stride_rank,
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
    grid = (min(num_segments, _WORKERS),)
    _cela_segment_owned_kernel[grid](
        input_ids,
        weights,
        output,
        batch_info.permutation,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        num_segments,
        weights.shape[0],
        input_ids.stride(0),
        batch_info.permutation.stride(0),
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        *weights.stride(),
        *output.stride(),
        BLOCK_RANK=128,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

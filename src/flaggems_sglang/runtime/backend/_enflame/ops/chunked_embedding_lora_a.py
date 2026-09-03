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
def _cela_gather_kernel_enflame(
    input_ids,
    weights,
    output,
    permutation,
    seg_ids,
    weight_indices,
    lora_ranks,
    total_tokens,
    ids_stride,
    perm_stride,
    segid_stride,
    widx_stride,
    ranks_stride,
    weight_stride_lora,
    weight_stride_rank,
    weight_stride_vocab,
    output_stride_token,
    output_stride_rank,
    BLOCK_RANK: tl.constexpr,
):
    # Enflame variant: int32 metadata everywhere, no explicit int64
    # casts (strides auto-specialize), no binary search and no early
    # returns - the segment id per position is precomputed on device
    # (searchsorted) so the kernel is straight-line gather only.
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for pos in range(pid, total_tokens, grid_size):
        seg = tl.load(seg_ids + pos * segid_stride)
        weight_index = tl.load(weight_indices + seg * widx_stride)
        rank = tl.load(lora_ranks + weight_index * ranks_stride)
        row = tl.load(permutation + pos * perm_stride)
        token_id = tl.load(input_ids + row * ids_stride)

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

    # Device-side route prep (vendor path): one searchsorted maps every
    # position to its owning segment, including across empty segments.
    seg_indptr = batch_info.seg_indptr
    positions = torch.arange(
        total_tokens, device=weights.device, dtype=seg_indptr.dtype
    )
    seg_ids = (torch.searchsorted(seg_indptr, positions, right=True) - 1).to(
        torch.int32
    )
    input_ids32 = (
        input_ids
        if input_ids.dtype == torch.int32
        else input_ids.to(torch.int32)
    )
    permutation32 = batch_info.permutation.to(torch.int32)
    weight_indices32 = batch_info.weight_indices.to(torch.int32)
    lora_ranks32 = batch_info.lora_ranks.to(torch.int32)

    grid = (min(total_tokens, _MAX_GRID),)
    _cela_gather_kernel_enflame[grid](
        input_ids32,
        weights,
        output,
        permutation32,
        seg_ids,
        weight_indices32,
        lora_ranks32,
        total_tokens,
        input_ids32.stride(0),
        permutation32.stride(0),
        seg_ids.stride(0),
        weight_indices32.stride(0),
        lora_ranks32.stride(0),
        *weights.stride(),
        *output.stride(),
        BLOCK_RANK=128,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

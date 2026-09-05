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

# Ascend vendor (FLA persistent): physical AI core grid with task
# stride — each persistent worker claims whole (segment, head_tile)
# tasks. The folding vendor held huawei at 2.12x; this structure gave
# T45 huawei 5.6x improvement on a similar gather+compute shape.

import torch
import triton
import triton.language as tl

_NUM_CORE = 32
_BLOCK_RANK = 128
_HEADS_TILE = 4


@triton.jit
def _cela_persistent_kernel(
    input_ids,
    weights,
    output,
    permutation,
    seg_indptr,
    weight_indices,
    lora_ranks,
    num_segments,
    num_lora,
    task_num,
    num_core,
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
    HEADS_TILE: tl.constexpr,
):
    # Each task = one (segment, head_tile) pair; head dimension is the
    # rank dimension [0, max_rank) split into tiles of HEADS_TILE*BLOCK_RANK
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        seg = task_id // HEADS_TILE
        ht = task_id - seg * HEADS_TILE

        if seg >= num_segments:
            continue

        start = tl.load(seg_indptr + seg * seg_stride)
        end = tl.load(seg_indptr + (seg + 1) * seg_stride)
        w_idx = tl.load(weight_indices + seg * widx_stride)
        w_idx = tl.minimum(w_idx, num_lora - 1)
        rank = tl.load(lora_ranks + w_idx * ranks_stride)
        if rank == 0:
            continue

        rank_lo = ht * BLOCK_RANK
        if rank_lo >= rank:
            continue
        rank_hi = tl.minimum(rank_lo + BLOCK_RANK, rank)

        w_idx64 = w_idx.to(tl.int64)
        for local in range(0, end - start):
            row = tl.load(permutation + (start + local) * perm_stride).to(tl.int64)
            token_id = tl.load(input_ids + row * ids_stride).to(tl.int64)

            for rb in range(0, 1):
                r_offs = rank_lo + tl.arange(0, BLOCK_RANK)
                r_mask = r_offs < rank_hi
                values = tl.load(
                    weights
                    + w_idx64 * weight_stride_lora
                    + r_offs * weight_stride_rank
                    + token_id * weight_stride_vocab,
                    mask=r_mask,
                    other=0.0,
                )
                tl.store(
                    output + row * output_stride_token + r_offs * output_stride_rank,
                    values,
                    mask=r_mask,
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

    heads_tile = max(1, max_rank // _BLOCK_RANK)
    task_num = num_segments * heads_tile
    grid = (_NUM_CORE,)
    _cela_persistent_kernel[grid](
        input_ids,
        weights,
        output,
        batch_info.permutation,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        num_segments,
        weights.shape[0],
        task_num,
        _NUM_CORE,
        input_ids.stride(0),
        batch_info.permutation.stride(0),
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        *weights.stride(),
        *output.stride(),
        BLOCK_RANK=_BLOCK_RANK,
        HEADS_TILE=heads_tile,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

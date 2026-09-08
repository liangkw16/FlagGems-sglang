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

# Ascend vendor (T17 E2a platform-proven form): 2D grid
# (token slots, batch) with token folding under the 65535 flatten cap;
# segment bounds and adapter metadata are read once per program and
# hoisted out of the token loop, rows guarded in-loop.

import torch
import triton
import triton.language as tl


@triton.jit
def _cela_fold_kernel_ascend(
    input_ids,
    weights,
    output,
    permutation,
    seg_indptr,
    weight_indices,
    lora_ranks,
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
    token_cap,
    BLOCK_RANK: tl.constexpr,
):
    token_slot = tl.program_id(0)
    batch_id = tl.program_id(1)
    start = tl.load(seg_indptr + batch_id * seg_stride)
    end = tl.load(seg_indptr + (batch_id + 1) * seg_stride)
    seg_len = end - start
    w_idx = tl.load(weight_indices + batch_id * widx_stride)
    # empty segments may carry sentinel adapter indices; clamp so the
    # ranks gather stays in bounds (the token loop never runs for them)
    w_idx = tl.minimum(w_idx, num_lora - 1)
    rank = tl.load(lora_ranks + w_idx * ranks_stride)
    w_idx64 = w_idx.to(tl.int64)
    num_rank_blocks = tl.cdiv(rank, BLOCK_RANK)
    for local in range(token_slot, seg_len, token_cap):
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
    bs = batch_info.bs
    if total_tokens == 0 or bs == 0:
        return output
    seg_indptr = batch_info.seg_indptr
    max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return output
    token_cap = max(1, min(max_len, 65535 // bs))
    grid = (token_cap, bs)
    _cela_fold_kernel_ascend[grid](
        input_ids,
        weights,
        output,
        batch_info.permutation,
        seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        weights.shape[0],
        input_ids.stride(0),
        batch_info.permutation.stride(0),
        seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        *weights.stride(),
        *output.stride(),
        token_cap,
        BLOCK_RANK=128,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

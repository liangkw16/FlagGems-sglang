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
# stride. No `continue` (unsupported in Triton) — validity folded into
# nested if blocks.

import torch
import triton
import triton.language as tl

_NUM_CORE = 32
_BLOCK_RANK = 128
_BLOCK_TOKENS = 8


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
    BLOCK_TOKENS: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        seg = task_id
        if seg < num_segments:
            start = tl.load(seg_indptr + seg * seg_stride)
            end = tl.load(seg_indptr + (seg + 1) * seg_stride)
            if start < end:
                w_idx = tl.load(weight_indices + seg * widx_stride)
                rank = tl.minimum(
                    tl.load(lora_ranks + w_idx * ranks_stride), MAX_RANK
                )
                w_idx64 = w_idx.to(tl.int64)
                for local in range(0, end - start, BLOCK_TOKENS):
                    offsets = local + tl.arange(0, BLOCK_TOKENS)
                    token_mask = offsets < end - start
                    rows = tl.load(
                        permutation + (start + offsets) * perm_stride,
                        token_mask,
                        0,
                    ).to(tl.int64)
                    tokens = tl.load(
                        input_ids + rows * ids_stride, token_mask, 0
                    ).to(tl.int64)
                    for rank_start in range(0, rank, BLOCK_RANK):
                        r = rank_start + tl.arange(0, BLOCK_RANK)
                        mask = token_mask[:, None] & (r[None, :] < rank)
                        values = tl.load(
                            weights
                            + w_idx64 * weight_stride_lora
                            + r[None, :] * weight_stride_rank
                            + tokens[:, None] * weight_stride_vocab,
                            mask,
                            0,
                        )
                        tl.store(
                            output
                            + rows[:, None] * output_stride_token
                            + r[None, :] * output_stride_rank,
                            values,
                            mask,
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

    task_num = num_segments
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
        BLOCK_TOKENS=_BLOCK_TOKENS,
        MAX_RANK=max_rank,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_embedding_lora_a"]

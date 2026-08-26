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
def _embedding_lora_a_kernel(
    input_ids,
    weights,
    output,
    extra_embeddings,
    vocab_size,
    input_stride,
    weight_stride_lora,
    weight_stride_rank,
    weight_stride_vocab,
    output_stride_token,
    output_stride_rank,
    extra_stride_lora,
    extra_stride_token,
    extra_stride_rank,
    seg_indptr,
    weight_indices,
    lora_ranks,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
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

    batch_id = tl.program_id(1)
    token_offset = tl.program_id(0)
    segment_start = tl.load(seg_indptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr + (batch_id + 1) * seg_indptr_stride)
    if token_offset >= segment_end - segment_start:
        return

    weight_index = tl.load(weight_indices + batch_id * weight_indices_stride)
    rank = tl.load(lora_ranks + weight_index * lora_ranks_stride)
    if rank == 0:
        return

    token_id = tl.load(
        input_ids + (segment_start + token_offset) * input_stride
    )
    num_rank_blocks = tl.cdiv(rank, BLOCK_RANK)

    for rank_block in range(num_rank_blocks):
        rank_offsets = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
        rank_mask = rank_offsets < rank
        is_extra = token_id >= vocab_size

        if HAS_EXTRA_EMBEDDINGS and is_extra:
            extra_token_id = token_id - vocab_size
            values = tl.load(
                extra_embeddings
                + weight_index * extra_stride_lora
                + extra_token_id * extra_stride_token
                + rank_offsets * extra_stride_rank,
                mask=rank_mask,
                other=0.0,
            )
        else:
            token_id = tl.minimum(token_id, vocab_size - 1)
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
            + (segment_start + token_offset) * output_stride_token
            + rank_offsets * output_stride_rank,
            values,
            mask=rank_mask,
        )


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

    if extra_embeddings is None:
        extra_embeddings = output
        extra_strides = (0, 0, 0)
    else:
        extra_strides = extra_embeddings.stride()

    grid = (batch_info.max_len, batch_info.bs)
    _embedding_lora_a_kernel[grid](
        input_ids,
        weights,
        output,
        extra_embeddings,
        vocab_size,
        input_ids.stride(0),
        *weights.stride(),
        *output.stride(),
        *extra_strides,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        BLOCK_RANK=128,
        HAS_EXTRA_EMBEDDINGS=extra_embeddings is not output,
        num_warps=2,
        num_stages=1,
    )
    return output


__all__ = ["embedding_lora_a"]

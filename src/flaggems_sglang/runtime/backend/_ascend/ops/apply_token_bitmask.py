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
def _apply_token_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    output_ptr,
    vocab_size,
    blocks_per_row,
    total_blocks,
    logits_stride_batch,
    logits_stride_vocab,
    bitmask_stride_batch,
    bitmask_stride_word,
    output_stride_batch,
    output_stride_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for logical_id in range(program_id, total_blocks, grid_size):
        batch = logical_id // blocks_per_row
        block = logical_id % blocks_per_row
        token = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        token_mask = token < vocab_size

        logits = tl.load(
            logits_ptr
            + batch * logits_stride_batch
            + token * logits_stride_vocab,
            mask=token_mask,
            other=0.0,
        )
        packed = tl.load(
            bitmask_ptr
            + batch * bitmask_stride_batch
            + (token // 32) * bitmask_stride_word,
            mask=token_mask,
            other=0,
        )
        allowed = ((packed >> (token % 32)) & 1) != 0
        tl.store(
            output_ptr
            + batch * output_stride_batch
            + token * output_stride_vocab,
            tl.where(allowed, logits, -float("inf")),
            mask=token_mask,
        )


def apply_token_bitmask(logits, bitmask):
    output = torch.empty_like(logits)
    if output.numel() == 0:
        return output

    batch_size, vocab_size = logits.shape
    block_size = 256
    blocks_per_row = triton.cdiv(vocab_size, block_size)
    total_blocks = batch_size * blocks_per_row
    grid = (min(total_blocks, 48),)
    _apply_token_bitmask_kernel[grid](
        logits,
        bitmask,
        output,
        vocab_size,
        blocks_per_row,
        total_blocks,
        logits.stride(0),
        logits.stride(1),
        bitmask.stride(0),
        bitmask.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["apply_token_bitmask"]

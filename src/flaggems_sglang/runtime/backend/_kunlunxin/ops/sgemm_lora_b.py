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

import triton
import triton.language as tl


@triton.jit
def _sgemm_lora_b_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    output_dim,
    x_stride_token,
    x_stride_rank,
    weight_stride_lora,
    weight_stride_output,
    weight_stride_rank,
    output_stride_token,
    output_stride_col,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
    scalings_stride,
    permutation_stride,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
    batch_id = tl.program_id(1)
    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    num_output_blocks = tl.cdiv(output_dim, BLOCK_N)
    matrix_pid = tl.program_id(0)
    token_block = matrix_pid // num_output_blocks
    output_block = matrix_pid % num_output_blocks
    if token_block * BLOCK_S >= segment_length:
        return

    weight_index = tl.load(
        weight_indices_ptr + batch_id * weight_indices_stride
    )
    rank = tl.load(lora_ranks_ptr + weight_index * lora_ranks_stride)
    if rank == 0:
        return

    offsets_s = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offsets_s < segment_length
    mask_n = offsets_n < output_dim
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr + (segment_start + offsets_s) * permutation_stride,
            mask=mask_s,
            other=0,
        )
    else:
        rows = segment_start + offsets_s

    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    offsets_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, RANK, BLOCK_K):
        k = k_start + offsets_k
        mask_k = k < RANK
        x = tl.load(
            x_ptr
            + rows[:, None] * x_stride_token
            + k[None, :] * x_stride_rank,
            mask=mask_s[:, None] & mask_k[None, :],
            other=0.0,
        )
        weights = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + offsets_n[None, :] * weight_stride_output
            + k[:, None] * weight_stride_rank,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x, weights, input_precision="ieee")

    output_offsets = (
        rows[:, None] * output_stride_token
        + offsets_n[None, :] * output_stride_col
    )
    output_mask = mask_s[:, None] & mask_n[None, :]
    base = tl.load(
        output_ptr + output_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    scaling = tl.load(scalings_ptr + weight_index * scalings_stride).to(
        tl.float32
    )
    tl.store(
        output_ptr + output_offsets,
        base + accumulator * scaling,
        mask=output_mask,
    )


def sgemm_lora_b(x, weights, batch_info, base_output):
    output = base_output.clone()
    if output.numel() == 0 or batch_info.bs == 0 or batch_info.max_len == 0:
        return output

    output_dim = weights.shape[1]
    rank = weights.shape[2]
    block_s = 32
    block_n = 32
    block_k = 32
    output_blocks = triton.cdiv(output_dim, block_n)
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * output_blocks,
        batch_info.bs,
    )
    permutation = batch_info.permutation
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        permutation if permutation is not None else batch_info.seg_indptr,
        output_dim,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        batch_info.scalings.stride(0),
        permutation.stride(0) if permutation is not None else 0,
        RANK=rank,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_PERMUTATION=permutation is not None,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["sgemm_lora_b"]

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
def _sgmv_expand_kernel_enflame(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    slice_offsets_ptr,
    num_lora,
    max_out_dim,
    x_stride_token,
    x_stride_rank,
    weight_stride_lora,
    weight_stride_output,
    weight_stride_rank,
    output_stride_token,
    output_stride_col,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Enflame variant: int32 metadata, no explicit int64 casts, and no
    # early returns (out-of-range blocks and rank-0 adapters fall out
    # through all-false store masks; sentinel adapter indices on empty
    # segments are clamped before the ranks/scalings gathers).
    batch_id = tl.program_id(2)
    slice_id = tl.program_id(1)
    segment_start = tl.load(seg_indptr_ptr + batch_id)
    segment_end = tl.load(seg_indptr_ptr + batch_id + 1)
    segment_length = segment_end - segment_start
    out_start = tl.load(slice_offsets_ptr + slice_id)
    out_end = tl.load(slice_offsets_ptr + slice_id + 1)
    output_size = out_end - out_start

    num_output_blocks = tl.cdiv(max_out_dim, BLOCK_N)
    matrix_pid = tl.program_id(0)
    token_block = matrix_pid // num_output_blocks
    output_block = matrix_pid - token_block * num_output_blocks

    weight_index = tl.load(weight_indices_ptr + batch_id)
    weight_index = tl.minimum(weight_index, num_lora - 1)
    rank = tl.load(lora_ranks_ptr + weight_index)
    rank_zero = rank == 0

    token_offsets = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = (token_offsets < segment_length) & (rank_zero == 0)
    output_mask = (output_offsets < output_size) & (
        output_block * BLOCK_N < output_size
    )
    rows = tl.load(
        permutation_ptr + segment_start + token_offsets,
        mask=token_offsets < segment_length,
        other=0,
    )

    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, RANK, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < RANK
        x = tl.load(
            x_ptr
            + rows[:, None] * x_stride_token
            + (slice_id * RANK + k[None, :]) * x_stride_rank,
            mask=(token_offsets[:, None] < segment_length) & k_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + (out_start + output_offsets[None, :]) * weight_stride_output
            + k[:, None] * weight_stride_rank,
            mask=k_mask[:, None] & (output_offsets[None, :] < output_size),
            other=0.0,
        )
        accumulator += tl.dot(x, weights, input_precision="ieee")

    mask = token_mask[:, None] & output_mask[None, :]
    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + (out_start + output_offsets[None, :]) * output_stride_col
    )
    base = tl.load(output_ptrs, mask=mask, other=0.0).to(tl.float32)
    scaling = tl.load(scalings_ptr + weight_index).to(tl.float32)
    tl.store(
        output_ptrs,
        (base + accumulator * scaling).to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    output = base_output.clone()
    n_slices = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if output.numel() == 0 or n_slices <= 0 or batch_info.bs == 0 or x.shape[0] == 0:
        return output

    seg_indptr = batch_info.seg_indptr.to(torch.int32)
    weight_indices = batch_info.weight_indices.to(torch.int32)
    lora_ranks = batch_info.lora_ranks.to(torch.int32)
    scalings = batch_info.scalings
    permutation = batch_info.permutation.to(torch.int32)
    slice_offsets32 = slice_offsets.to(torch.int32)
    max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return output

    block_s = 64
    block_n = 128
    block_k = 32
    output_blocks = triton.cdiv(int(max_slice_size), block_n)
    grid = (
        triton.cdiv(max_len, block_s) * output_blocks,
        n_slices,
        batch_info.bs,
    )
    _sgmv_expand_kernel_enflame[grid](
        x,
        weights,
        output,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        permutation,
        slice_offsets32,
        weights.shape[0],
        int(max_slice_size),
        *x.stride(),
        *weights.stride(),
        *output.stride(),
        RANK=rank,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["chunked_sgmv_expand"]

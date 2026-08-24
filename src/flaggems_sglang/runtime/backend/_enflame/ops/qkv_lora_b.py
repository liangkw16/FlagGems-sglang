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
def _qkv_lora_b_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    output_offset_ptr,
    max_qkv_out_dim,
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
    output_offset_stride,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_rank = tl.cast(x_stride_rank, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_output = tl.cast(weight_stride_output, tl.int64)
    weight_stride_rank = tl.cast(weight_stride_rank, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    batch_id = tl.program_id(2)
    slice_id = tl.program_id(1)
    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    output_start = tl.load(output_offset_ptr + slice_id * output_offset_stride)
    output_end = tl.load(
        output_offset_ptr + (slice_id + 1) * output_offset_stride
    )
    output_size = output_end - output_start

    num_output_blocks = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    matrix_pid = tl.program_id(0)
    token_block = matrix_pid // num_output_blocks
    output_block = matrix_pid - token_block * num_output_blocks
    if token_block * BLOCK_S >= segment_length:
        return
    if output_block * BLOCK_N >= output_size:
        return

    weight_index = tl.load(
        weight_indices_ptr + batch_id * weight_indices_stride
    )
    if tl.load(lora_ranks_ptr + weight_index * lora_ranks_stride) == 0:
        return

    token_offsets = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < segment_length
    output_mask = output_offsets < output_size
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr
            + (segment_start + token_offsets) * permutation_stride,
            mask=token_mask,
            other=0,
        )
    else:
        rows = segment_start + token_offsets

    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, RANK, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < RANK
        x = tl.load(
            x_ptr
            + rows[:, None] * x_stride_token
            + (slice_id * RANK + k[None, :]) * x_stride_rank,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + (output_start + output_offsets[None, :]) * weight_stride_output
            + k[:, None] * weight_stride_rank,
            mask=k_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        if x.dtype == tl.float32:
            x_hi = x.to(tl.float16)
            x_lo = (x - x_hi.to(tl.float32)).to(tl.float16)
            w_hi = weights.to(tl.float16)
            w_lo = (weights - w_hi.to(tl.float32)).to(tl.float16)
            accumulator += tl.dot(x_hi, w_hi)
            accumulator += tl.dot(x_hi, w_lo)
            accumulator += tl.dot(x_lo, w_hi)
        else:
            accumulator += tl.dot(x, weights)

    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + (output_start + output_offsets[None, :]) * output_stride_col
    )
    mask = token_mask[:, None] & output_mask[None, :]
    base = tl.load(output_ptrs, mask=mask, other=0.0).to(tl.float32)
    scaling = tl.load(scalings_ptr + weight_index * scalings_stride).to(
        tl.float32
    )
    tl.store(output_ptrs, base + accumulator * scaling, mask=mask)


def qkv_lora_b(
    x,
    qkv_lora_b,
    batch_info,
    output_offset,
    max_qkv_out_dim,
    base_output,
):
    output = base_output.clone()
    n_slices = output_offset.numel() - 1
    rank = qkv_lora_b.shape[-1]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if (
        output.numel() == 0
        or n_slices <= 0
        or max_qkv_out_dim <= 0
        or batch_info.bs == 0
        or batch_info.max_len == 0
    ):
        return output

    block_s = 64
    block_n = 128
    block_k = 32
    output_blocks = triton.cdiv(max_qkv_out_dim, block_n)
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * output_blocks,
        n_slices,
        batch_info.bs,
    )
    permutation = batch_info.permutation
    _qkv_lora_b_kernel[grid](
        x,
        qkv_lora_b,
        output,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        permutation if permutation is not None else batch_info.seg_indptr,
        output_offset,
        max_qkv_out_dim,
        *x.stride(),
        *qkv_lora_b.stride(),
        *output.stride(),
        batch_info.seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        batch_info.scalings.stride(0),
        permutation.stride(0) if permutation is not None else 0,
        output_offset.stride(0),
        RANK=rank,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_PERMUTATION=permutation is not None,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["qkv_lora_b"]

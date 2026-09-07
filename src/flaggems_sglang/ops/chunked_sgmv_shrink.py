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
def _sgmv_shrink_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    max_out_dim,
    x_stride_token,
    x_stride_col,
    weight_stride_lora,
    weight_stride_out,
    weight_stride_k,
    output_stride_token,
    output_stride_col,
    seg_indptr_stride,
    weight_indices_stride,
    permutation_stride,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_col = tl.cast(x_stride_col, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_out = tl.cast(weight_stride_out, tl.int64)
    weight_stride_k = tl.cast(weight_stride_k, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    batch_id = tl.program_id(2)
    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    weight_index = tl.load(weight_indices_ptr + batch_id * weight_indices_stride)
    if weight_index < 0:
        return
    if segment_start >= segment_end:
        return

    num_output_blocks = tl.cdiv(max_out_dim, BLOCK_N)
    matrix_pid = tl.program_id(0)
    token_block = matrix_pid // num_output_blocks
    output_block = matrix_pid - token_block * num_output_blocks
    if token_block * BLOCK_S >= segment_length:
        return

    token_offsets = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < segment_length
    output_mask = output_offsets < max_out_dim
    rows = tl.load(
        permutation_ptr + (segment_start + token_offsets) * permutation_stride,
        mask=token_mask,
        other=0,
    )

    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, RANK, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < RANK
        x = tl.load(
            x_ptr + rows[:, None] * x_stride_token + k[None, :] * x_stride_col,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + output_offsets[None, :] * weight_stride_out
            + k[:, None] * weight_stride_k,
            mask=k_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x, weights, input_precision="ieee")

    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + output_offsets[None, :] * output_stride_col
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=token_mask[:, None] & output_mask[None, :],
    )


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    output = torch.zeros(S, N, dtype=x.dtype, device=x.device)
    if output.numel() == 0 or batch_info.bs == 0:
        return output

    seg_indptr = batch_info.seg_indptr
    max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return output

    # e4: shape-adaptive token tile (SGLang production premise: request
    # segments are short, so BM=64 pads 4x on a <=16-row segment). Only
    # BLOCK_S adapts; BLOCK_N/K, warps and stages stay at the E6
    # platform-proven values, and the (token_tile, output_tile) grid
    # keeps full coverage of segments longer than one tile.
    if max_len <= 16:
        block_s = 16
    elif max_len <= 32:
        block_s = 32
    else:
        block_s = 64
    block_n = 128
    block_k = 32
    grid = (
        triton.cdiv(max_len, block_s) * triton.cdiv(N, block_n),
        1,
        batch_info.bs,
    )  # num_slices axis removed: kernel never reads program_id(1)
    _sgmv_shrink_kernel[grid](
        x,
        weights,
        output,
        seg_indptr,
        batch_info.weight_indices,
        batch_info.permutation,
        N,
        *x.stride(),
        *weights.stride(),
        *output.stride(),
        seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.permutation.stride(0),
        RANK=K,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return output


__all__ = ["chunked_sgmv_shrink"]

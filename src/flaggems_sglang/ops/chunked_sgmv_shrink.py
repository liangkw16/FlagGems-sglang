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
    max_out_dim: tl.constexpr,
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
    TOTAL_TOKENS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOKEN_WORKERS: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_col = tl.cast(x_stride_col, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_out = tl.cast(weight_stride_out, tl.int64)
    weight_stride_k = tl.cast(weight_stride_k, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    num_output_blocks = tl.cdiv(max_out_dim, BLOCK_N)
    batch_id = tl.program_id(0) // num_output_blocks
    output_block = tl.program_id(0) % num_output_blocks
    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    weight_index = tl.load(
        weight_indices_ptr + batch_id * weight_indices_stride
    )
    if weight_index < 0:
        return
    if segment_start >= segment_end:
        return

    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < max_out_dim
    token_start = tl.program_id(1) * BLOCK_S
    # Stripe the whole segment across workers; average length only tunes launch.
    if token_start >= segment_length:
        return
    for _ in range(tl.cdiv(TOTAL_TOKENS, TOKEN_WORKERS * BLOCK_S)):
        token_offsets = token_start + tl.arange(0, BLOCK_S)
        token_mask = token_offsets < segment_length
        rows = tl.load(
            permutation_ptr
            + (segment_start + token_offsets) * permutation_stride,
            mask=token_mask,
            other=0,
        )

        accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        correction = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        k_offsets = tl.arange(0, BLOCK_K)
        for k_start in range(0, RANK, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < RANK
            x = tl.load(
                x_ptr
                + rows[:, None] * x_stride_token
                + k[None, :] * x_stride_col,
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
            partial = tl.dot(x, weights, input_precision="ieee")
            if x_ptr.dtype.element_ty == tl.float32 and RANK >= 1024:
                # Compensate long-K summation near zero at the FP32 tolerance.
                adjusted = partial - correction
                updated = accumulator + adjusted
                correction = (updated - accumulator) - adjusted
                accumulator = updated
            else:
                accumulator += partial

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
        token_start += TOKEN_WORKERS * BLOCK_S


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    output = torch.zeros(S, N, dtype=x.dtype, device=x.device)
    if output.numel() == 0 or batch_info.bs == 0:
        return output

    seg_indptr = batch_info.seg_indptr
    average_length = triton.cdiv(S, batch_info.bs)
    block_s = (
        16 if average_length <= 16 else 32 if average_length <= 32 else 64
    )
    block_n = 64 if x.dtype == torch.float32 else 128
    block_k = 32
    token_workers = min(32, triton.cdiv(S, block_s))
    # Keep the first worker of adjacent segments adjacent in launch order.
    grid = (triton.cdiv(N, block_n) * batch_info.bs, token_workers)
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
        TOTAL_TOKENS=S,
        RANK=K,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        TOKEN_WORKERS=token_workers,
        num_warps=4,
        num_stages=3,
    )
    return output


__all__ = ["chunked_sgmv_shrink"]

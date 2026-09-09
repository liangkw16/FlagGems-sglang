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
    TOTAL_ROWS: tl.constexpr,
    PART_K: tl.constexpr,
    SPLIT: tl.constexpr,
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
    weight_index = tl.load(
        weight_indices_ptr + batch_id * weight_indices_stride
    )
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

    part = tl.program_id(1) if SPLIT else 0
    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(
        part * PART_K, tl.minimum((part + 1) * PART_K, RANK), BLOCK_K
    ):
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
        + part * TOTAL_ROWS * max_out_dim
        + rows[:, None] * output_stride_token
        + output_offsets[None, :] * output_stride_col
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=token_mask[:, None] & output_mask[None, :],
    )


@triton.jit
def _shrink_merge(
    parts, output, SIZE: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for part in range(PARTS):
        value += tl.load(
            parts + part * SIZE + offsets, mask=offsets < SIZE, other=0.0
        )
    tl.store(output + offsets, value, mask=offsets < SIZE)


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

    # Narrow-rank GEMM has few output tiles. Partition the long input
    # dimension, then deterministically merge FP32 partials (Punica SGMV).
    if x.dtype == torch.float32:
        block_s = 16 if max_len <= 16 else 32
        block_n = min(128, max(16, triton.next_power_of_2(N)))
    else:
        # Native low-precision GEMM already fills Tensor Cores; retain the
        # measured baseline schedule instead of paying split-K overhead.
        block_s = 16 if max_len <= 16 else 32 if max_len <= 32 else 64
        block_n = 128
    block_k = 32
    tiles = (
        triton.cdiv(max_len, block_s) * triton.cdiv(N, block_n) * batch_info.bs
    )
    parts = (
        4
        if x.dtype == torch.float32 and K >= 1024 and N <= 128 and tiles < 128
        else 1
    )
    part_k = triton.cdiv(triton.cdiv(K, block_k), parts) * block_k
    target = (
        torch.zeros((parts, S, N), dtype=torch.float32, device=x.device)
        if parts > 1
        else output
    )
    grid = (
        triton.cdiv(max_len, block_s) * triton.cdiv(N, block_n),
        parts,
        batch_info.bs,
    )
    _sgmv_shrink_kernel[grid](
        x,
        weights,
        target,
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
        TOTAL_ROWS=S,
        PART_K=part_k,
        SPLIT=(parts > 1),
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    if parts > 1:
        _shrink_merge[(triton.cdiv(S * N, 256),)](
            target,
            output,
            S * N,
            parts,
            BLOCK=256,
            num_warps=4,
        )
    return output


__all__ = ["chunked_sgmv_shrink"]

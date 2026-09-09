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
def _segment_tile_prefix(
    seg,
    prefix,
    B: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ids = tl.arange(0, BLOCK)
    start = tl.load(seg + ids * STRIDE, mask=ids < B, other=0)
    end = tl.load(seg + (ids + 1) * STRIDE, mask=ids < B, other=0)
    counts = tl.cdiv(end - start, BM)
    cumulative = tl.cumsum(counts, axis=0)
    tl.store(prefix + ids + 1, cumulative, mask=ids < B)
    tl.store(prefix, 0)


@triton.jit
def _chunked_sgmv_expand_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    slice_offsets_ptr,
    max_out_dim,
    prefix_ptr,
    task_offset,
    COMPACT: tl.constexpr,
    NSLICES: tl.constexpr,
    BATCHES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
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
    slice_offsets_stride,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_rank = tl.cast(x_stride_rank, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_output = tl.cast(weight_stride_output, tl.int64)
    weight_stride_rank = tl.cast(weight_stride_rank, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    num_output_blocks = tl.cdiv(max_out_dim, BLOCK_N)
    if COMPACT:
        task = task_offset + tl.program_id(0)
        output_block = task % num_output_blocks
        slice_id = (task // num_output_blocks) % NSLICES
        row_tile = task // (num_output_blocks * NSLICES)
        actual_tiles = tl.load(prefix_ptr + BATCHES)
        if row_tile >= actual_tiles:
            return
        lo = 0
        hi = BATCHES + 1
        for _ in tl.static_range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            boundary = tl.load(prefix_ptr + tl.minimum(mid, BATCHES))
            right = (lo < hi) & (boundary <= row_tile)
            hi = tl.where((lo < hi) & ~right, mid, hi)
            lo = tl.where(right, mid + 1, lo)
        batch_id = lo - 1
        token_block = row_tile - tl.load(prefix_ptr + batch_id)
    else:
        batch_id = tl.program_id(2)
        slice_id = tl.program_id(1)
        token_block = tl.program_id(0) // num_output_blocks
        output_block = tl.program_id(0) % num_output_blocks
    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    out_start = tl.load(slice_offsets_ptr + slice_id * slice_offsets_stride)
    out_end = tl.load(
        slice_offsets_ptr + (slice_id + 1) * slice_offsets_stride
    )
    output_size = out_end - out_start

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
            x_ptr
            + rows[:, None] * x_stride_token
            + (slice_id * RANK + k[None, :]) * x_stride_rank,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + (out_start + output_offsets[None, :]) * weight_stride_output
            + k[:, None] * weight_stride_rank,
            mask=k_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x, weights, input_precision="ieee")

    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + (out_start + output_offsets[None, :]) * output_stride_col
    )
    mask = token_mask[:, None] & output_mask[None, :]
    base = tl.load(output_ptrs, mask=mask, other=0.0).to(tl.float32)
    scaling = tl.load(scalings_ptr + weight_index * scalings_stride).to(
        tl.float32
    )
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
    if (
        output.numel() == 0
        or n_slices <= 0
        or batch_info.bs == 0
        or x.shape[0] == 0
    ):
        return output

    seg_indptr = batch_info.seg_indptr
    block_s = 64
    block_n = 128
    block_k = 32
    max_len = getattr(batch_info, "max_len", None)
    # ponytail: one prefix CTA handles up to 4096 segments; retain the
    # existing maximum-length dispatch above that capacity.
    compact = max_len is None and batch_info.bs <= 4096
    output_blocks = triton.cdiv(int(max_slice_size), block_n)
    if compact:
        prefix = torch.empty(
            (batch_info.bs + 1,), dtype=torch.int32, device=x.device
        )
        _segment_tile_prefix[(1,)](
            seg_indptr,
            prefix,
            batch_info.bs,
            seg_indptr.stride(0),
            block_s,
            BLOCK=triton.next_power_of_2(batch_info.bs),
            num_warps=4,
        )
        # sum(ceil(segment_len/BM)) <= floor((S+B*(BM-1))/BM).
        row_tiles = (
            batch_info.permutation.numel() + batch_info.bs * (block_s - 1)
        ) // block_s
        tasks = row_tiles * n_slices * output_blocks
    else:
        if max_len is None:
            max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
        if max_len == 0:
            return output
        prefix = seg_indptr
        tasks = 1
    for start in range(0, tasks, 65535):
        grid = (
            (min(tasks - start, 65535),)
            if compact
            else (
                triton.cdiv(max_len, block_s) * output_blocks,
                n_slices,
                batch_info.bs,
            )
        )
        _chunked_sgmv_expand_kernel[grid](
            x,
            weights,
            output,
            seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            batch_info.permutation,
            slice_offsets,
            int(max_slice_size),
            prefix,
            start,
            compact,
            n_slices,
            batch_info.bs,
            (batch_info.bs + 1).bit_length(),
            *x.stride(),
            *weights.stride(),
            *output.stride(),
            seg_indptr.stride(0),
            batch_info.weight_indices.stride(0),
            batch_info.lora_ranks.stride(0),
            batch_info.scalings.stride(0),
            batch_info.permutation.stride(0),
            slice_offsets.stride(0),
            RANK=rank,
            BLOCK_S=block_s,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
    return output


__all__ = ["chunked_sgmv_expand"]

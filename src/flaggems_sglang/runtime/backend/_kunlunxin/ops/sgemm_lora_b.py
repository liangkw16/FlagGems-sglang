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

_MAX_GRID_SIZE = 65535


@triton.jit
def _pack_x_kernel(
    x_ptr,
    packed_x_ptr,
    seg_indptr_ptr,
    permutation_ptr,
    max_len,
    program_start,
    x_stride_token,
    x_stride_rank,
    seg_indptr_stride,
    permutation_stride,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_rank = tl.cast(x_stride_rank, tl.int64)

    rank_tiles = tl.cdiv(RANK, BLOCK_K)
    tiles_per_batch = tl.cdiv(max_len, BLOCK_M) * rank_tiles
    logical_id = program_start + tl.program_id(0)
    batch_id = logical_id // tiles_per_batch
    matrix_id = logical_id - batch_id * tiles_per_batch
    token_tile = matrix_id // rank_tiles
    rank_tile = matrix_id - token_tile * rank_tiles

    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    token_offsets = token_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rank_offsets = rank_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    token_mask = token_offsets < segment_length
    rank_mask = rank_offsets < RANK
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr
            + (segment_start + token_offsets) * permutation_stride,
            mask=token_mask,
            other=0,
        )
    else:
        rows = segment_start + token_offsets

    values = tl.load(
        x_ptr
        + rows[:, None] * x_stride_token
        + rank_offsets[None, :] * x_stride_rank,
        mask=token_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    packed_offsets = (
        batch_id * max_len + token_offsets[:, None]
    ) * RANK + rank_offsets[None, :]
    tl.store(
        packed_x_ptr + packed_offsets,
        values,
        mask=(token_offsets[:, None] < max_len) & rank_mask[None, :],
    )


@triton.jit
def _safe_adapter_kernel(
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    safe_adapter_ptr,
    batch_size,
    batch_start,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_ids = (
        batch_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    )
    batch_mask = batch_ids < batch_size
    segment_start = tl.load(
        seg_indptr_ptr + batch_ids * seg_indptr_stride,
        mask=batch_mask,
        other=0,
    )
    segment_end = tl.load(
        seg_indptr_ptr + (batch_ids + 1) * seg_indptr_stride,
        mask=batch_mask,
        other=0,
    )
    nonempty_mask = batch_mask & (segment_start != segment_end)
    weight_index = tl.load(
        weight_indices_ptr + batch_ids * weight_indices_stride,
        mask=nonempty_mask,
        other=0,
    )
    lora_rank = tl.load(
        lora_ranks_ptr + weight_index * lora_ranks_stride,
        mask=nonempty_mask,
        other=0,
    )
    safe_adapter = tl.where(nonempty_mask & (lora_rank != 0), weight_index, 0)
    tl.store(
        safe_adapter_ptr + batch_ids,
        safe_adapter,
        mask=batch_mask,
    )


@triton.jit
def _regular_bmm_kernel(
    packed_x_ptr,
    transposed_weights_ptr,
    safe_adapter_ptr,
    products_ptr,
    max_len,
    output_dim,
    program_start,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_tiles = tl.cdiv(output_dim, BLOCK_N)
    tiles_per_batch = tl.cdiv(max_len, BLOCK_M) * output_tiles
    logical_id = program_start + tl.program_id(0)
    batch_id = logical_id // tiles_per_batch
    matrix_id = logical_id - batch_id * tiles_per_batch
    token_tile = matrix_id // output_tiles
    output_tile = matrix_id - token_tile * output_tiles
    token_offsets = token_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    weight_index = tl.load(safe_adapter_ptr + batch_id)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rank_offsets = tl.arange(0, BLOCK_K)

    for rank_start in range(0, RANK, BLOCK_K):
        rank = rank_start + rank_offsets
        packed_x = tl.load(
            packed_x_ptr
            + (batch_id * max_len + token_offsets[:, None]) * RANK
            + rank[None, :],
            mask=(token_offsets[:, None] < max_len) & (rank[None, :] < RANK),
            other=0.0,
        )
        transposed_weights = tl.load(
            transposed_weights_ptr
            + (weight_index * RANK + rank[:, None]) * output_dim
            + output_offsets[None, :],
            mask=(output_offsets[None, :] < output_dim)
            & (rank[:, None] < RANK),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.dot(
            packed_x, transposed_weights, input_precision="ieee"
        )

    product_offsets = (
        batch_id * max_len + token_offsets[:, None]
    ) * output_dim + output_offsets[None, :]
    tl.store(
        products_ptr + product_offsets,
        accumulator,
        mask=(token_offsets[:, None] < max_len)
        & (output_offsets[None, :] < output_dim),
    )


@triton.jit
def _scatter_add_kernel(
    products_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    max_len,
    output_dim,
    program_start,
    output_stride_token,
    output_stride_col,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
    scalings_stride,
    permutation_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    output_tiles = tl.cdiv(output_dim, BLOCK_N)
    tiles_per_batch = tl.cdiv(max_len, BLOCK_M) * output_tiles
    logical_id = program_start + tl.program_id(0)
    batch_id = logical_id // tiles_per_batch
    matrix_id = logical_id - batch_id * tiles_per_batch
    token_tile = matrix_id // output_tiles
    output_tile = matrix_id - token_tile * output_tiles

    segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
    segment_length = segment_end - segment_start
    token_offsets = token_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    if token_tile * BLOCK_M >= segment_length:
        return
    weight_index = tl.load(
        weight_indices_ptr + batch_id * weight_indices_stride
    )
    if tl.load(lora_ranks_ptr + weight_index * lora_ranks_stride) == 0:
        return

    output_offsets = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < segment_length
    output_mask = output_offsets < output_dim
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr
            + (segment_start + token_offsets) * permutation_stride,
            mask=token_mask,
            other=0,
        )
    else:
        rows = segment_start + token_offsets

    product_offsets = (
        batch_id * max_len + token_offsets[:, None]
    ) * output_dim + output_offsets[None, :]
    products = tl.load(
        products_ptr + product_offsets,
        mask=token_mask[:, None] & output_mask[None, :],
        other=0.0,
    )
    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + output_offsets[None, :] * output_stride_col
    )
    mask = token_mask[:, None] & output_mask[None, :]
    base = tl.load(output_ptrs, mask=mask, other=0.0).to(tl.float32)
    scaling = tl.load(scalings_ptr + weight_index * scalings_stride).to(
        tl.float32
    )
    tl.store(output_ptrs, base + products * scaling, mask=mask)


def sgemm_lora_b(x, weights, batch_info, base_output):
    output = base_output.clone()
    if (
        output.numel() == 0
        or weights.shape[0] == 0
        or weights.shape[2] == 0
        or batch_info.bs == 0
        or batch_info.max_len == 0
    ):
        return output

    batch_size = batch_info.bs
    max_len = batch_info.max_len
    output_dim = weights.shape[1]
    rank = weights.shape[2]
    block_m = 32
    block_n = 32
    block_k = 32
    safe_adapter_block = 256
    packed_x = torch.empty(
        (batch_size, max_len, rank), dtype=torch.float32, device=x.device
    )
    transposed_weights = weights.transpose(1, 2).contiguous()
    safe_adapters = torch.empty(
        (batch_size,),
        dtype=torch.int32,
        device=batch_info.weight_indices.device,
    )
    products = torch.empty(
        (batch_size, max_len, output_dim),
        dtype=torch.float32,
        device=x.device,
    )
    permutation = batch_info.permutation
    permutation_arg = (
        permutation if permutation is not None else batch_info.seg_indptr
    )

    safe_adapter_span = _MAX_GRID_SIZE * safe_adapter_block
    for batch_start in range(0, batch_size, safe_adapter_span):
        batch_count = min(safe_adapter_span, batch_size - batch_start)
        grid = (triton.cdiv(batch_count, safe_adapter_block),)
        _safe_adapter_kernel[grid](
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            safe_adapters,
            batch_size,
            batch_start,
            batch_info.seg_indptr.stride(0),
            batch_info.weight_indices.stride(0),
            batch_info.lora_ranks.stride(0),
            BLOCK_SIZE=safe_adapter_block,
            num_warps=4,
            num_stages=1,
        )

    rank_tiles = triton.cdiv(rank, block_k)
    pack_x_programs = batch_size * triton.cdiv(max_len, block_m) * rank_tiles
    for program_start in range(0, pack_x_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, pack_x_programs - program_start),)
        _pack_x_kernel[grid](
            x,
            packed_x,
            batch_info.seg_indptr,
            permutation_arg,
            max_len,
            program_start,
            *x.stride(),
            batch_info.seg_indptr.stride(0),
            permutation.stride(0) if permutation is not None else 0,
            RANK=rank,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            HAS_PERMUTATION=permutation is not None,
            num_warps=4,
            num_stages=1,
        )

    bmm_programs = (
        batch_size
        * triton.cdiv(max_len, block_m)
        * triton.cdiv(output_dim, block_n)
    )
    for program_start in range(0, bmm_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, bmm_programs - program_start),)
        _regular_bmm_kernel[grid](
            packed_x,
            transposed_weights,
            safe_adapters,
            products,
            max_len,
            output_dim,
            program_start,
            RANK=rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
        _scatter_add_kernel[grid](
            products,
            output,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            permutation_arg,
            max_len,
            output_dim,
            program_start,
            *output.stride(),
            batch_info.seg_indptr.stride(0),
            batch_info.weight_indices.stride(0),
            batch_info.lora_ranks.stride(0),
            batch_info.scalings.stride(0),
            permutation.stride(0) if permutation is not None else 0,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HAS_PERMUTATION=permutation is not None,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["sgemm_lora_b"]

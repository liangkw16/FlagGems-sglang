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
def _pack_sequences_kernel(
    B_ptr,
    x_ptr,
    dt_ptr,
    dA_ptr,
    cu_seqlens_ptr,
    packed_x_ptr,
    packed_B_ptr,
    headdim,
    dstate,
    total_seqlen,
    chunk_size,
    packed_seq_len,
    packed_feature,
    nheads,
    head_group_ratio,
    feature_tiles,
    program_start,
    stride_B_seqlen,
    stride_B_group,
    stride_B_dstate,
    stride_x_seqlen,
    stride_x_head,
    stride_x_headdim,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dA_head,
    stride_dA_chunk,
    stride_dA_csize,
    stride_cu,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    stride_B_seqlen = tl.cast(stride_B_seqlen, tl.int64)
    stride_B_group = tl.cast(stride_B_group, tl.int64)
    stride_B_dstate = tl.cast(stride_B_dstate, tl.int64)
    stride_x_seqlen = tl.cast(stride_x_seqlen, tl.int64)
    stride_x_head = tl.cast(stride_x_head, tl.int64)
    stride_x_headdim = tl.cast(stride_x_headdim, tl.int64)
    stride_dt_head = tl.cast(stride_dt_head, tl.int64)
    stride_dt_chunk = tl.cast(stride_dt_chunk, tl.int64)
    stride_dt_csize = tl.cast(stride_dt_csize, tl.int64)
    stride_dA_head = tl.cast(stride_dA_head, tl.int64)
    stride_dA_chunk = tl.cast(stride_dA_chunk, tl.int64)
    stride_dA_csize = tl.cast(stride_dA_csize, tl.int64)
    stride_cu = tl.cast(stride_cu, tl.int64)

    k_tiles = tl.cdiv(packed_seq_len, BLOCK_K)
    tiles_per_head = k_tiles * feature_tiles
    logical_id = program_start + tl.program_id(0)
    batch_head = logical_id // tiles_per_head
    tile = logical_id - batch_head * tiles_per_head
    batch = batch_head // nheads
    head = batch_head - batch * nheads
    k_tile = tile // feature_tiles
    feature_tile = tile - k_tile * feature_tiles
    group = head // head_group_ratio

    start = tl.load(cu_seqlens_ptr + batch * stride_cu).to(tl.int64)
    end = tl.load(cu_seqlens_ptr + (batch + 1) * stride_cu).to(tl.int64)
    sequence_length = end - start
    safe_end = tl.maximum(end, 1)
    chunk = (safe_end - 1) // chunk_size
    chunk_start = chunk * chunk_size
    start_relative = start - chunk_start
    end_relative = safe_end - chunk_start
    slice_start = tl.where(
        start_relative < 0,
        tl.maximum(start_relative + chunk_size, 0),
        tl.minimum(start_relative, chunk_size),
    )
    slice_stop = tl.minimum(tl.maximum(end_relative, 0), chunk_size)
    scale_length = tl.maximum(slice_stop - slice_start, 0)

    offsets_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_d = feature_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = offsets_k < sequence_length
    scale_k = tl.where(scale_length == 1, 0, offsets_k)
    scale_mask = token_mask & (scale_k < scale_length)
    relative = slice_start + scale_k
    dA_base = head * stride_dA_head + chunk * stride_dA_chunk
    dA_last = tl.load(
        dA_ptr + dA_base + (end_relative - 1) * stride_dA_csize
    ).to(tl.float32)
    safe_relative = tl.minimum(relative, chunk_size - 1)
    dt = tl.load(
        dt_ptr
        + head * stride_dt_head
        + chunk * stride_dt_chunk
        + safe_relative * stride_dt_csize
    ).to(tl.float32)
    dA = tl.load(dA_ptr + dA_base + safe_relative * stride_dA_csize).to(
        tl.float32
    )
    scale = tl.where(scale_mask, tl.exp(dA_last - dA) * dt, 0.0)
    token = start + offsets_k
    safe_token = tl.minimum(token, total_seqlen - 1)
    safe_x_d = tl.minimum(offsets_d, headdim - 1)

    x = tl.load(
        x_ptr
        + safe_token[:, None] * stride_x_seqlen
        + head * stride_x_head
        + safe_x_d[None, :] * stride_x_headdim
    ).to(tl.float32)
    x = tl.where(token_mask[:, None] & (offsets_d[None, :] < headdim), x, 0.0)
    packed_x_offsets = (
        batch_head * packed_seq_len + offsets_k[:, None]
    ) * packed_feature + offsets_d[None, :]
    tl.store(packed_x_ptr + packed_x_offsets, x)

    safe_B_d = tl.minimum(offsets_d, dstate - 1)
    B = tl.load(
        B_ptr
        + safe_token[:, None] * stride_B_seqlen
        + group * stride_B_group
        + safe_B_d[None, :] * stride_B_dstate
    ).to(tl.float32)
    B = tl.where(
        token_mask[:, None] & (offsets_d[None, :] < dstate),
        B * scale[:, None],
        0.0,
    )
    packed_B_offsets = (
        batch_head * packed_seq_len + offsets_k[:, None]
    ) * packed_feature + offsets_d[None, :]
    tl.store(packed_B_ptr + packed_B_offsets, B)


@triton.jit
def _regular_bmm_kernel(
    packed_x_ptr,
    packed_B_ptr,
    output_ptr,
    padded_headdim,
    padded_dstate,
    packed_feature,
    program_start,
    PACKED_SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tiles = padded_dstate // BLOCK_N
    tiles_per_head = (padded_headdim // BLOCK_M) * n_tiles
    logical_id = program_start + tl.program_id(0)
    batch_head = logical_id // tiles_per_head
    tile = logical_id - batch_head * tiles_per_head
    tile_m = tile // n_tiles
    tile_n = tile - tile_m * n_tiles
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for block_start in range(0, PACKED_SEQ_LEN, BLOCK_K):
        current_k = block_start + offsets_k
        x = tl.load(
            packed_x_ptr
            + (batch_head * PACKED_SEQ_LEN + current_k[None, :])
            * packed_feature
            + offsets_m[:, None]
        )
        B = tl.load(
            packed_B_ptr
            + (batch_head * PACKED_SEQ_LEN + current_k[:, None])
            * packed_feature
            + offsets_n[None, :]
        )
        accumulator += tl.dot(x, B, input_precision="ieee")

    output_offsets = (
        batch_head * padded_headdim * padded_dstate
        + offsets_m[:, None] * padded_dstate
        + offsets_n[None, :]
    )
    tl.store(output_ptr + output_offsets, accumulator)


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    if B.ndim != 3 or x.ndim != 3 or dt.ndim != 3:
        raise ValueError("B, x, and dt must be three-dimensional")
    if dA_cumsum.shape != dt.shape:
        raise ValueError("dA_cumsum must have the same shape as dt")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
        raise ValueError("cu_seqlens must be a nonempty vector")

    total_seqlen, nheads, headdim = x.shape
    B_seqlen, ngroups, dstate = B.shape
    dt_heads, nchunks, chunk_size = dt.shape
    if B_seqlen != total_seqlen:
        raise ValueError("B and x must have the same packed sequence length")
    if dt_heads != nheads:
        raise ValueError("dt and x must have the same number of heads")
    if ngroups <= 0 or nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by a positive ngroups")
    if chunk_size <= 0 or nchunks <= 0:
        raise ValueError("dt must contain positive chunk dimensions")
    if total_seqlen > nchunks * chunk_size:
        raise ValueError("dt does not contain enough chunks for packed tokens")

    batch = cu_seqlens.numel() - 1
    output_shape = (batch, nheads, headdim, dstate)
    if batch == 0 or nheads == 0 or headdim == 0 or dstate == 0:
        return torch.empty(
            output_shape, dtype=chunk_states.dtype, device=x.device
        )

    block_m, block_n, block_k = 32, 32, 32
    pack_block_k = 16
    pack_block_d = 16
    batch_heads = batch * nheads
    max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    packed_seq_len = triton.cdiv(max_seq_len, block_k) * block_k
    packed_feature = triton.cdiv(max(headdim, dstate), block_m) * block_m
    padded_headdim = triton.cdiv(headdim, block_m) * block_m
    padded_dstate = triton.cdiv(dstate, block_n) * block_n
    feature_tiles = packed_feature // pack_block_d
    packed_x = torch.empty(
        (batch, nheads, packed_seq_len, packed_feature),
        dtype=torch.float32,
        device=x.device,
    )
    packed_B = torch.empty(
        (batch, nheads, packed_seq_len, packed_feature),
        dtype=torch.float32,
        device=x.device,
    )
    output_storage = torch.empty(
        (batch, nheads, padded_headdim, padded_dstate),
        dtype=chunk_states.dtype,
        device=x.device,
    )

    pack_programs = (
        batch_heads * (packed_seq_len // pack_block_k) * feature_tiles
        if total_seqlen > 0
        else 0
    )
    for program_start in range(0, pack_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, pack_programs - program_start),)
        _pack_sequences_kernel[grid](
            B,
            x,
            dt,
            dA_cumsum,
            cu_seqlens,
            packed_x,
            packed_B,
            headdim,
            dstate,
            total_seqlen,
            chunk_size,
            packed_seq_len,
            packed_feature,
            nheads,
            nheads // ngroups,
            feature_tiles,
            program_start,
            *B.stride(),
            *x.stride(),
            *dt.stride(),
            *dA_cumsum.stride(),
            cu_seqlens.stride(0),
            BLOCK_K=pack_block_k,
            BLOCK_D=pack_block_d,
            num_warps=4,
            num_stages=1,
        )

    tiles_per_head = (padded_headdim // block_m) * (padded_dstate // block_n)
    bmm_programs = batch_heads * tiles_per_head
    for program_start in range(0, bmm_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, bmm_programs - program_start),)
        _regular_bmm_kernel[grid](
            packed_x,
            packed_B,
            output_storage,
            padded_headdim,
            padded_dstate,
            packed_feature,
            program_start,
            PACKED_SEQ_LEN=packed_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )

    return output_storage[:, :, :headdim, :dstate]


__all__ = ["chunk_state_varlen"]

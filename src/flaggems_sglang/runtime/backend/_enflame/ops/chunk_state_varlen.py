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
    chunk_size,
    max_seq_len,
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
    k_tiles = tl.cdiv(max_seq_len, BLOCK_K)
    tiles_per_head = k_tiles * feature_tiles
    logical_id = program_start + tl.program_id(0)
    batch_head = logical_id // tiles_per_head
    tile = logical_id - batch_head * tiles_per_head
    batch = batch_head // nheads
    head = batch_head - batch * nheads
    k_tile = tile // feature_tiles
    feature_tile = tile - k_tile * feature_tiles
    group = head // head_group_ratio

    start = tl.load(cu_seqlens_ptr + batch * stride_cu)
    end = tl.load(cu_seqlens_ptr + (batch + 1) * stride_cu)
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
        dA_ptr + dA_base + (end_relative - 1) * stride_dA_csize,
        mask=sequence_length > 0,
        other=0.0,
    ).to(tl.float32)
    dt = tl.load(
        dt_ptr
        + head * stride_dt_head
        + chunk * stride_dt_chunk
        + relative * stride_dt_csize,
        mask=scale_mask,
        other=0.0,
    ).to(tl.float32)
    dA = tl.load(
        dA_ptr + dA_base + relative * stride_dA_csize,
        mask=scale_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.where(scale_mask, tl.exp(dA_last - dA) * dt, 0.0)
    token = start + offsets_k

    x = tl.load(
        x_ptr
        + token[:, None] * stride_x_seqlen
        + head * stride_x_head
        + offsets_d[None, :] * stride_x_headdim,
        mask=token_mask[:, None] & (offsets_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    packed_x_offsets = (
        batch_head * max_seq_len + offsets_k[:, None]
    ) * headdim + offsets_d[None, :]
    tl.store(
        packed_x_ptr + packed_x_offsets,
        x,
        mask=(offsets_k[:, None] < max_seq_len)
        & (offsets_d[None, :] < headdim),
    )

    B = tl.load(
        B_ptr
        + token[:, None] * stride_B_seqlen
        + group * stride_B_group
        + offsets_d[None, :] * stride_B_dstate,
        mask=token_mask[:, None] & (offsets_d[None, :] < dstate),
        other=0.0,
    ).to(tl.float32)
    packed_B_offsets = (
        batch_head * max_seq_len + offsets_k[:, None]
    ) * dstate + offsets_d[None, :]
    tl.store(
        packed_B_ptr + packed_B_offsets,
        B * scale[:, None],
        mask=(offsets_k[:, None] < max_seq_len)
        & (offsets_d[None, :] < dstate),
    )


@triton.jit
def _regular_bmm_kernel(
    packed_x_ptr,
    packed_B_ptr,
    output_ptr,
    headdim,
    dstate,
    program_start,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tiles = tl.cdiv(dstate, BLOCK_N)
    tiles_per_head = tl.cdiv(headdim, BLOCK_M) * n_tiles
    logical_id = program_start + tl.program_id(0)
    batch_head = logical_id // tiles_per_head
    tile = logical_id - batch_head * tiles_per_head
    tile_m = tile // n_tiles
    tile_n = tile - tile_m * n_tiles
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for block_start in range(0, MAX_SEQ_LEN, BLOCK_K):
        current_k = block_start + offsets_k
        x = tl.load(
            packed_x_ptr
            + (batch_head * MAX_SEQ_LEN + current_k[None, :]) * headdim
            + offsets_m[:, None],
            mask=(offsets_m[:, None] < headdim)
            & (current_k[None, :] < MAX_SEQ_LEN),
            other=0.0,
        )
        B = tl.load(
            packed_B_ptr
            + (batch_head * MAX_SEQ_LEN + current_k[:, None]) * dstate
            + offsets_n[None, :],
            mask=(current_k[:, None] < MAX_SEQ_LEN)
            & (offsets_n[None, :] < dstate),
            other=0.0,
        )
        accumulator += tl.dot(x, B, input_precision="ieee")

    output_offsets = (
        batch_head * headdim * dstate
        + offsets_m[:, None] * dstate
        + offsets_n[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(offsets_m[:, None] < headdim) & (offsets_n[None, :] < dstate),
    )


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
    output = torch.empty(
        (batch, nheads, headdim, dstate),
        dtype=chunk_states.dtype,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    routed_cu_seqlens = (
        cu_seqlens.to(torch.int32)
        if cu_seqlens.dtype == torch.int64
        else cu_seqlens
    )

    block_m, block_n, block_k = 32, 32, 32
    pack_block_k = 32
    pack_block_d = 32
    batch_heads = batch * nheads
    max_seq_len = int(
        (routed_cu_seqlens[1:] - routed_cu_seqlens[:-1]).max().item()
    )
    feature_tiles = triton.cdiv(max(headdim, dstate), pack_block_d)
    packed_x = torch.empty(
        (batch, nheads, max_seq_len, headdim),
        dtype=torch.float32,
        device=x.device,
    )
    packed_B = torch.empty(
        (batch, nheads, max_seq_len, dstate),
        dtype=torch.float32,
        device=x.device,
    )

    pack_programs = (
        batch_heads * triton.cdiv(max_seq_len, pack_block_k) * feature_tiles
    )
    for program_start in range(0, pack_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, pack_programs - program_start),)
        _pack_sequences_kernel[grid](
            B,
            x,
            dt,
            dA_cumsum,
            routed_cu_seqlens,
            packed_x,
            packed_B,
            headdim,
            dstate,
            chunk_size,
            max_seq_len,
            nheads,
            nheads // ngroups,
            feature_tiles,
            program_start,
            *B.stride(),
            *x.stride(),
            *dt.stride(),
            *dA_cumsum.stride(),
            routed_cu_seqlens.stride(0),
            BLOCK_K=pack_block_k,
            BLOCK_D=pack_block_d,
            num_warps=4,
            num_stages=1,
        )

    tiles_per_head = triton.cdiv(headdim, block_m) * triton.cdiv(
        dstate, block_n
    )
    bmm_programs = batch_heads * tiles_per_head
    for program_start in range(0, bmm_programs, _MAX_GRID_SIZE):
        grid = (min(_MAX_GRID_SIZE, bmm_programs - program_start),)
        _regular_bmm_kernel[grid](
            packed_x,
            packed_B,
            output,
            headdim,
            dstate,
            program_start,
            MAX_SEQ_LEN=max_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )

    return output


__all__ = ["chunk_state_varlen"]

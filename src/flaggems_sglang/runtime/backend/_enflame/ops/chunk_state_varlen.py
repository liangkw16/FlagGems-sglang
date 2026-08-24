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
def _chunk_state_varlen_kernel(
    B_ptr,
    x_ptr,
    dt_ptr,
    dA_ptr,
    cu_seqlens_ptr,
    output_ptr,
    headdim,
    dstate,
    chunk_size,
    head_group_ratio,
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
    stride_output_batch,
    stride_output_head,
    stride_output_headdim,
    stride_output_dstate,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    stride_output_batch = tl.cast(stride_output_batch, tl.int64)
    stride_output_head = tl.cast(stride_output_head, tl.int64)
    stride_output_headdim = tl.cast(stride_output_headdim, tl.int64)
    stride_output_dstate = tl.cast(stride_output_dstate, tl.int64)

    tile = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    n_tiles = tl.cdiv(dstate, BLOCK_N)
    tile_m = tile // n_tiles
    tile_n = tile % n_tiles
    group = head // head_group_ratio

    start = tl.load(cu_seqlens_ptr + batch * stride_cu).to(tl.int64)
    end = tl.load(cu_seqlens_ptr + (batch + 1) * stride_cu).to(tl.int64)
    sequence_length = end - start
    chunk = (end - 1) // chunk_size
    chunk_start = chunk * chunk_size
    start_relative = start - chunk_start
    end_relative = end - chunk_start
    slice_start = tl.where(
        start_relative < 0,
        tl.maximum(start_relative + chunk_size, 0),
        tl.minimum(start_relative, chunk_size),
    )
    slice_stop = tl.minimum(tl.maximum(end_relative, 0), chunk_size)
    scale_length = tl.maximum(slice_stop - slice_start, 0)

    dA_base = head * stride_dA_head + chunk * stride_dA_chunk
    dA_last = tl.load(
        dA_ptr + dA_base + (end_relative - 1) * stride_dA_csize,
        mask=sequence_length > 0,
        other=0.0,
    ).to(tl.float32)

    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for block_start in range(0, sequence_length, BLOCK_K):
        current_k = block_start + offsets_k
        k_mask = current_k < sequence_length
        token = start + current_k
        scale_k = tl.where(scale_length == 1, 0, current_k)
        scale_mask = k_mask & (scale_k < scale_length)
        relative = slice_start + scale_k

        x = tl.load(
            x_ptr
            + token[None, :] * stride_x_seqlen
            + head * stride_x_head
            + offsets_m[:, None] * stride_x_headdim,
            mask=(offsets_m[:, None] < headdim) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        B = tl.load(
            B_ptr
            + token[:, None] * stride_B_seqlen
            + group * stride_B_group
            + offsets_n[None, :] * stride_B_dstate,
            mask=k_mask[:, None] & (offsets_n[None, :] < dstate),
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
        accumulator += tl.dot(x, B * scale[:, None], input_precision="ieee")

    output_offsets = (
        batch * stride_output_batch
        + head * stride_output_head
        + offsets_m[:, None] * stride_output_headdim
        + offsets_n[None, :] * stride_output_dstate
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

    block_m = 32
    block_n = 32
    grid = (
        triton.cdiv(headdim, block_m) * triton.cdiv(dstate, block_n),
        batch,
        nheads,
    )
    _chunk_state_varlen_kernel[grid](
        B,
        x,
        dt,
        dA_cumsum,
        cu_seqlens,
        output,
        headdim,
        dstate,
        chunk_size,
        nheads // ngroups,
        *B.stride(),
        *x.stride(),
        *dt.stride(),
        *dA_cumsum.stride(),
        cu_seqlens.stride(0),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_state_varlen"]

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
def _chunk_state_kernel(
    B_ptr,
    x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    output_ptr,
    headdim,
    dstate,
    chunk_size,
    nchunks,
    head_group_ratio,
    stride_B_batch,
    stride_B_seqlen,
    stride_B_group,
    stride_B_dstate,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_headdim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dA_batch,
    stride_dA_head,
    stride_dA_chunk,
    stride_dA_csize,
    stride_output_batch,
    stride_output_chunk,
    stride_output_head,
    stride_output_headdim,
    stride_output_dstate,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    stride_B_batch = tl.cast(stride_B_batch, tl.int64)
    stride_B_seqlen = tl.cast(stride_B_seqlen, tl.int64)
    stride_B_group = tl.cast(stride_B_group, tl.int64)
    stride_B_dstate = tl.cast(stride_B_dstate, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_seqlen = tl.cast(stride_x_seqlen, tl.int64)
    stride_x_head = tl.cast(stride_x_head, tl.int64)
    stride_x_headdim = tl.cast(stride_x_headdim, tl.int64)
    stride_dt_batch = tl.cast(stride_dt_batch, tl.int64)
    stride_dt_head = tl.cast(stride_dt_head, tl.int64)
    stride_dt_chunk = tl.cast(stride_dt_chunk, tl.int64)
    stride_dt_csize = tl.cast(stride_dt_csize, tl.int64)
    stride_dA_batch = tl.cast(stride_dA_batch, tl.int64)
    stride_dA_head = tl.cast(stride_dA_head, tl.int64)
    stride_dA_chunk = tl.cast(stride_dA_chunk, tl.int64)
    stride_dA_csize = tl.cast(stride_dA_csize, tl.int64)
    stride_output_batch = tl.cast(stride_output_batch, tl.int64)
    stride_output_chunk = tl.cast(stride_output_chunk, tl.int64)
    stride_output_head = tl.cast(stride_output_head, tl.int64)
    stride_output_headdim = tl.cast(stride_output_headdim, tl.int64)
    stride_output_dstate = tl.cast(stride_output_dstate, tl.int64)

    tile_id = tl.program_id(0)
    batch_chunk_id = tl.program_id(1)
    head_id = tl.program_id(2)
    batch_id = batch_chunk_id // nchunks
    chunk_id = batch_chunk_id - batch_id * nchunks
    group_id = head_id // head_group_ratio
    n_tiles = tl.cdiv(dstate, BLOCK_N)
    m_tile = tile_id // n_tiles
    n_tile = tile_id - m_tile * n_tiles

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    sequence_base = chunk_id * chunk_size
    B_base = batch_id * stride_B_batch + group_id * stride_B_group
    x_base = batch_id * stride_x_batch + head_id * stride_x_head
    dt_base = (
        batch_id * stride_dt_batch
        + head_id * stride_dt_head
        + chunk_id * stride_dt_chunk
    )
    dA_base = (
        batch_id * stride_dA_batch
        + head_id * stride_dA_head
        + chunk_id * stride_dA_chunk
    )
    dA_last = tl.load(
        dA_cumsum_ptr + dA_base + (chunk_size - 1) * stride_dA_csize
    ).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(chunk_size, BLOCK_K)):
        current_k = k_block * BLOCK_K + k_offsets
        sequence_offsets = sequence_base + current_k
        k_mask = current_k < chunk_size
        x = tl.load(
            x_ptr
            + x_base
            + sequence_offsets[None, :] * stride_x_seqlen
            + m_offsets[:, None] * stride_x_headdim,
            mask=(m_offsets[:, None] < headdim) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        B = tl.load(
            B_ptr
            + B_base
            + sequence_offsets[:, None] * stride_B_seqlen
            + n_offsets[None, :] * stride_B_dstate,
            mask=k_mask[:, None] & (n_offsets[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            dt_ptr + dt_base + current_k * stride_dt_csize,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        dA = tl.load(
            dA_cumsum_ptr + dA_base + current_k * stride_dA_csize,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.where(k_mask, tl.exp(dA_last - dA) * dt, 0.0)
        B *= scale[:, None]
        accumulator += tl.dot(x, B, input_precision="ieee")

    output_offsets = (
        batch_id * stride_output_batch
        + chunk_id * stride_output_chunk
        + head_id * stride_output_head
        + m_offsets[:, None] * stride_output_headdim
        + n_offsets[None, :] * stride_output_dstate
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(m_offsets[:, None] < headdim) & (n_offsets[None, :] < dstate),
    )


def chunk_state(B, x, dt, dA_cumsum):
    if B.ndim != 4 or x.ndim != 4 or dt.ndim != 4:
        raise ValueError("B, x, and dt must be four-dimensional")
    if dA_cumsum.shape != dt.shape:
        raise ValueError("dA_cumsum must have the same shape as dt")

    batch, seqlen, nheads, headdim = x.shape
    B_batch, B_seqlen, ngroups, dstate = B.shape
    dt_batch, dt_heads, nchunks, chunk_size = dt.shape
    if (B_batch, B_seqlen) != (batch, seqlen):
        raise ValueError("B batch and sequence dimensions must match x")
    if (dt_batch, dt_heads) != (batch, nheads):
        raise ValueError("dt batch and head dimensions must match x")
    if ngroups <= 0 or nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by a positive ngroups")
    if chunk_size <= 0 or seqlen != nchunks * chunk_size:
        raise ValueError("seqlen must equal nchunks * chunk_size")

    output = torch.empty(
        (batch, nchunks, nheads, headdim, dstate),
        dtype=torch.float32,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    block_m = 32
    block_n = 32
    grid = (
        triton.cdiv(headdim, block_m) * triton.cdiv(dstate, block_n),
        batch * nchunks,
        nheads,
    )
    _chunk_state_kernel[grid](
        B,
        x,
        dt,
        dA_cumsum,
        output,
        headdim,
        dstate,
        chunk_size,
        nchunks,
        nheads // ngroups,
        *B.stride(),
        *x.stride(),
        *dt.stride(),
        *dA_cumsum.stride(),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_state"]

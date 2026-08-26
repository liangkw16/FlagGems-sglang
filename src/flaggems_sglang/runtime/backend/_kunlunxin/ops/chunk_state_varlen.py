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
    dA_cumsum_ptr,
    output_ptr,
    batch_id,
    start,
    sequence_length,
    chunk,
    slice_start,
    last_relative,
    headdim,
    dstate,
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
    stride_output_batch,
    stride_output_head,
    stride_output_headdim,
    stride_output_dstate,
    BROADCAST_SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
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
    stride_output_batch = tl.cast(stride_output_batch, tl.int64)
    stride_output_head = tl.cast(stride_output_head, tl.int64)
    stride_output_headdim = tl.cast(stride_output_headdim, tl.int64)
    stride_output_dstate = tl.cast(stride_output_dstate, tl.int64)

    tile_id = tl.program_id(0)
    head_id = tl.program_id(1)
    group_id = head_id // head_group_ratio
    n_tiles = tl.cdiv(dstate, BLOCK_N)
    m_tile = tile_id // n_tiles
    n_tile = tile_id - m_tile * n_tiles

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    B_base = group_id * stride_B_group
    x_base = head_id * stride_x_head
    dt_base = head_id * stride_dt_head + chunk * stride_dt_chunk
    dA_base = head_id * stride_dA_head + chunk * stride_dA_chunk
    dA_last = tl.load(
        dA_cumsum_ptr + dA_base + last_relative * stride_dA_csize
    ).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for block_start in range(0, MAX_SEQ_LEN, BLOCK_K):
        current_k = block_start + k_offsets
        k_mask = current_k < sequence_length
        token = start + current_k

        x = tl.load(
            x_ptr
            + x_base
            + token[None, :] * stride_x_seqlen
            + m_offsets[:, None] * stride_x_headdim,
            mask=(m_offsets[:, None] < headdim) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        B = tl.load(
            B_ptr
            + B_base
            + token[:, None] * stride_B_seqlen
            + n_offsets[None, :] * stride_B_dstate,
            mask=k_mask[:, None] & (n_offsets[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        if BROADCAST_SCALE:
            dt = tl.load(dt_ptr + dt_base + slice_start * stride_dt_csize).to(
                tl.float32
            )
            dA = tl.load(
                dA_cumsum_ptr + dA_base + slice_start * stride_dA_csize
            ).to(tl.float32)
        else:
            relative = slice_start + current_k
            dt = tl.load(
                dt_ptr + dt_base + relative * stride_dt_csize,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            dA = tl.load(
                dA_cumsum_ptr + dA_base + relative * stride_dA_csize,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
        scale = tl.where(k_mask, tl.exp(dA_last - dA) * dt, 0.0)
        B *= scale[:, None]
        accumulator += tl.dot(x, B, input_precision="ieee")

    output_offsets = (
        batch_id * stride_output_batch
        + head_id * stride_output_head
        + m_offsets[:, None] * stride_output_headdim
        + n_offsets[None, :] * stride_output_dstate
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(m_offsets[:, None] < headdim) & (n_offsets[None, :] < dstate),
    )


def _sequence_plan(start, end, chunk_size):
    sequence_length = end - start
    if sequence_length <= 0:
        return sequence_length, 0, 0, 0

    chunk = (end - 1) // chunk_size
    chunk_start = chunk * chunk_size
    start_relative = start - chunk_start
    end_relative = end - chunk_start
    if start_relative < 0:
        slice_start = max(start_relative + chunk_size, 0)
    else:
        slice_start = min(start_relative, chunk_size)
    slice_stop = min(max(end_relative, 0), chunk_size)
    scale_length = max(slice_stop - slice_start, 0)
    return sequence_length, chunk, slice_start, scale_length


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

    boundaries = [int(value) for value in cu_seqlens.tolist()]
    plans = [
        _sequence_plan(start, end, chunk_size)
        for start, end in zip(boundaries, boundaries[1:])
    ]
    max_seq_len = max((plan[0] for plan in plans), default=0)
    output = torch.zeros(
        output_shape, dtype=chunk_states.dtype, device=x.device
    )
    if max_seq_len <= 0:
        return output

    block_m = 32
    block_n = 32
    block_k = 64 if max_seq_len >= 256 else 32
    padded_seq_len = triton.cdiv(max_seq_len, block_k) * block_k
    grid = (
        triton.cdiv(headdim, block_m) * triton.cdiv(dstate, block_n),
        nheads,
    )
    for batch_id, ((start, _), plan) in enumerate(
        zip(zip(boundaries, boundaries[1:]), plans)
    ):
        sequence_length, chunk, slice_start, scale_length = plan
        if sequence_length <= 0:
            continue
        if scale_length not in (1, sequence_length):
            raise ValueError(
                "the reference scale must broadcast over the sequence"
            )

        _chunk_state_varlen_kernel[grid](
            B,
            x,
            dt,
            dA_cumsum,
            output,
            batch_id,
            start,
            sequence_length,
            chunk,
            slice_start,
            slice_start + scale_length - 1,
            headdim,
            dstate,
            nheads // ngroups,
            *B.stride(),
            *x.stride(),
            *dt.stride(),
            *dA_cumsum.stride(),
            *output.stride(),
            BROADCAST_SCALE=scale_length == 1,
            MAX_SEQ_LEN=padded_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["chunk_state_varlen"]

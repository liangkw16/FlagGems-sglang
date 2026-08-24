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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_cumsum_kernel(
    dt_ptr,
    a_ptr,
    bias_ptr,
    dt_out_ptr,
    da_cumsum_ptr,
    seqlen,
    nheads,
    dt_stride_b,
    dt_stride_s,
    dt_stride_h,
    a_stride,
    bias_stride,
    out_stride_b,
    out_stride_h,
    out_stride_c,
    out_stride_s,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SOFTPLUS: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offsets_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_c = tl.arange(0, BLOCK_SIZE)
    offsets_s = pid_chunk * CHUNK_SIZE + offsets_c
    mask_h = offsets_h < nheads
    mask_s = (offsets_c < CHUNK_SIZE) & (offsets_s < seqlen)
    mask = mask_h[:, None] & mask_s[None, :]

    dt_offsets = (
        pid_batch * dt_stride_b
        + offsets_s[None, :] * dt_stride_s
        + offsets_h[:, None] * dt_stride_h
    )
    values = tl.load(dt_ptr + dt_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offsets_h * bias_stride,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        values += bias[:, None]
    if SOFTPLUS:
        values = tl.where(values <= 20.0, tl.log(1.0 + tl.exp(values)), values)
    values = tl.where(values < 0.0, 0.0, values)
    values = tl.where(mask, values, 0.0)

    a = tl.load(a_ptr + offsets_h * a_stride, mask=mask_h, other=0.0).to(
        tl.float32
    )
    da_cumsum = tl.cumsum(values * a[:, None], axis=1)
    out_offsets = (
        pid_batch * out_stride_b
        + offsets_h[:, None] * out_stride_h
        + pid_chunk * out_stride_c
        + offsets_c[None, :] * out_stride_s
    )
    output_mask = mask_h[:, None] & (offsets_c[None, :] < CHUNK_SIZE)
    tl.store(dt_out_ptr + out_offsets, values, mask=output_mask)
    tl.store(da_cumsum_ptr + out_offsets, da_cumsum, mask=output_mask)


@triton.jit
def _chunk_cumsum_folded_kernel(
    dt_ptr,
    a_ptr,
    bias_ptr,
    dt_out_ptr,
    da_cumsum_ptr,
    seqlen,
    nheads,
    nchunks,
    total_programs,
    dt_stride_b,
    dt_stride_s,
    dt_stride_h,
    a_stride,
    bias_stride,
    out_stride_b,
    out_stride_h,
    out_stride_c,
    out_stride_s,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SOFTPLUS: tl.constexpr,
):
    program_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    head_blocks = tl.cdiv(nheads, BLOCK_H)
    for logical_id in range(program_id, total_programs, grid_size):
        pid_batch = logical_id // (nchunks * head_blocks)
        remainder = logical_id - pid_batch * (nchunks * head_blocks)
        pid_chunk = remainder // head_blocks
        pid_h = remainder - pid_chunk * head_blocks
        offsets_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offsets_c = tl.arange(0, BLOCK_SIZE)
        offsets_s = pid_chunk * CHUNK_SIZE + offsets_c
        mask_h = offsets_h < nheads
        mask_s = (offsets_c < CHUNK_SIZE) & (offsets_s < seqlen)
        mask = mask_h[:, None] & mask_s[None, :]

        dt_offsets = (
            pid_batch * dt_stride_b
            + offsets_s[None, :] * dt_stride_s
            + offsets_h[:, None] * dt_stride_h
        )
        values = tl.load(dt_ptr + dt_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offsets_h * bias_stride,
                mask=mask_h,
                other=0.0,
            ).to(tl.float32)
            values += bias[:, None]
        if SOFTPLUS:
            values = tl.where(
                values <= 20.0, tl.log(1.0 + tl.exp(values)), values
            )
        values = tl.where(values < 0.0, 0.0, values)
        values = tl.where(mask, values, 0.0)

        a = tl.load(a_ptr + offsets_h * a_stride, mask=mask_h, other=0.0).to(
            tl.float32
        )
        da_cumsum = tl.cumsum(values * a[:, None], axis=1)
        out_offsets = (
            pid_batch * out_stride_b
            + offsets_h[:, None] * out_stride_h
            + pid_chunk * out_stride_c
            + offsets_c[None, :] * out_stride_s
        )
        output_mask = mask_h[:, None] & (offsets_c[None, :] < CHUNK_SIZE)
        tl.store(dt_out_ptr + out_offsets, values, mask=output_mask)
        tl.store(da_cumsum_ptr + out_offsets, da_cumsum, mask=output_mask)


def chunk_cumsum(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    output_shape = (batch, nheads, nchunks, chunk_size)
    allocate = torch.zeros if seqlen % chunk_size else torch.empty
    dt_out = allocate(output_shape, dtype=torch.float32, device=dt.device)
    da_cumsum = allocate(output_shape, dtype=torch.float32, device=dt.device)
    if batch == 0 or seqlen == 0 or nheads == 0:
        return dt_out, da_cumsum

    block_size = triton.next_power_of_2(chunk_size)
    block_h = max(1, min(8, 4096 // block_size))
    head_blocks = triton.cdiv(nheads, block_h)
    total_programs = head_blocks * nchunks * batch
    if total_programs <= 65535:
        grid = (head_blocks, nchunks, batch)
        _chunk_cumsum_kernel[grid](
            dt,
            A,
            dt_bias if dt_bias is not None else A,
            dt_out,
            da_cumsum,
            seqlen,
            nheads,
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else A.stride(0),
            dt_out.stride(0),
            dt_out.stride(1),
            dt_out.stride(2),
            dt_out.stride(3),
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=block_size,
            BLOCK_H=block_h,
            HAS_BIAS=dt_bias is not None,
            SOFTPLUS=dt_softplus,
            num_warps=4,
            num_stages=1,
        )
        return dt_out, da_cumsum
    grid = (min(total_programs, 4096),)
    _chunk_cumsum_folded_kernel[grid](
        dt,
        A,
        dt_bias if dt_bias is not None else A,
        dt_out,
        da_cumsum,
        seqlen,
        nheads,
        nchunks,
        total_programs,
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else A.stride(0),
        dt_out.stride(0),
        dt_out.stride(1),
        dt_out.stride(2),
        dt_out.stride(3),
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
        BLOCK_H=block_h,
        HAS_BIAS=dt_bias is not None,
        SOFTPLUS=dt_softplus,
        num_warps=4,
        num_stages=1,
    )
    return dt_out, da_cumsum


__all__ = ["chunk_cumsum"]

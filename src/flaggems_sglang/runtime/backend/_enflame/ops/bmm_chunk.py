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
def _bmm_chunk_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    chunk_size,
    k_size,
    ngroups,
    cg_count,
    total_programs,
    a_stride_batch,
    a_stride_seqlen,
    a_stride_group,
    a_stride_k,
    b_stride_batch,
    b_stride_seqlen,
    b_stride_group,
    b_stride_k,
    output_stride_batch,
    output_stride_chunk,
    output_stride_group,
    output_stride_m,
    output_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INPUT_DTYPE: tl.constexpr,
):
    a_stride_batch = tl.cast(a_stride_batch, tl.int64)
    a_stride_seqlen = tl.cast(a_stride_seqlen, tl.int64)
    a_stride_group = tl.cast(a_stride_group, tl.int64)
    a_stride_k = tl.cast(a_stride_k, tl.int64)
    b_stride_batch = tl.cast(b_stride_batch, tl.int64)
    b_stride_seqlen = tl.cast(b_stride_seqlen, tl.int64)
    b_stride_group = tl.cast(b_stride_group, tl.int64)
    b_stride_k = tl.cast(b_stride_k, tl.int64)
    output_stride_batch = tl.cast(output_stride_batch, tl.int64)
    output_stride_chunk = tl.cast(output_stride_chunk, tl.int64)
    output_stride_group = tl.cast(output_stride_group, tl.int64)
    output_stride_m = tl.cast(output_stride_m, tl.int64)
    output_stride_n = tl.cast(output_stride_n, tl.int64)

    program_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_n_tiles = tl.cdiv(chunk_size, BLOCK_N)
    num_m_tiles = tl.cdiv(chunk_size, BLOCK_M)
    tiles_per_cg = num_m_tiles * num_n_tiles
    for logical_id in range(program_id, total_programs, grid_size):
        batch_id = logical_id // (cg_count * tiles_per_cg)
        remainder = logical_id - batch_id * (cg_count * tiles_per_cg)
        chunk_group_id = remainder // tiles_per_cg
        tile_id = remainder - chunk_group_id * tiles_per_cg
        chunk_id = chunk_group_id // ngroups
        group_id = chunk_group_id - chunk_id * ngroups
        m_tile = tile_id // num_n_tiles
        n_tile = tile_id - m_tile * num_n_tiles

        m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        a_base = (
            batch_id * a_stride_batch
            + chunk_id * chunk_size * a_stride_seqlen
            + group_id * a_stride_group
        )
        b_base = (
            batch_id * b_stride_batch
            + chunk_id * chunk_size * b_stride_seqlen
            + group_id * b_stride_group
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            current_k = k_block * BLOCK_K + k_offsets
            a = tl.load(
                a_ptr
                + a_base
                + m_offsets[:, None] * a_stride_seqlen
                + current_k[None, :] * a_stride_k,
                mask=(m_offsets[:, None] < chunk_size)
                & (current_k[None, :] < k_size),
                other=0.0,
            )
            b = tl.load(
                b_ptr
                + b_base
                + current_k[:, None] * b_stride_k
                + n_offsets[None, :] * b_stride_seqlen,
                mask=(current_k[:, None] < k_size)
                & (n_offsets[None, :] < chunk_size),
                other=0.0,
            )
            if not USE_INPUT_DTYPE:
                a_hi = a.to(tl.float16)
                a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
                b_hi = b.to(tl.float16)
                b_lo = (b - b_hi.to(tl.float32)).to(tl.float16)
                accumulator += tl.dot(a_hi, b_hi)
                accumulator += tl.dot(a_hi, b_lo)
                accumulator += tl.dot(a_lo, b_hi)
            else:
                accumulator += tl.dot(a, b)

        output_offsets = (
            batch_id * output_stride_batch
            + chunk_id * output_stride_chunk
            + group_id * output_stride_group
            + m_offsets[:, None] * output_stride_m
            + n_offsets[None, :] * output_stride_n
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=(m_offsets[:, None] < chunk_size)
            & (n_offsets[None, :] < chunk_size),
        )


def bmm_chunk(a, b, chunk_size, causal=False):
    if a.ndim != 4 or b.shape != a.shape:
        raise ValueError("a and b must have the same four-dimensional shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch, seqlen, ngroups, k_size = a.shape
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")

    nchunks = seqlen // chunk_size
    output = torch.empty(
        (batch, nchunks, ngroups, chunk_size, chunk_size),
        dtype=torch.float32,
        device=a.device,
    )
    if output.numel() == 0:
        return output

    block_m = 64
    block_n = 64
    cg_count = nchunks * ngroups
    total_programs = (
        triton.cdiv(chunk_size, block_m)
        * triton.cdiv(chunk_size, block_n)
        * batch
        * cg_count
    )
    grid = (min(total_programs, 6),)
    _bmm_chunk_kernel[grid](
        a,
        b,
        output,
        chunk_size,
        k_size,
        ngroups,
        cg_count,
        total_programs,
        *a.stride(),
        *b.stride(),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        USE_INPUT_DTYPE=a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16),
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["bmm_chunk"]

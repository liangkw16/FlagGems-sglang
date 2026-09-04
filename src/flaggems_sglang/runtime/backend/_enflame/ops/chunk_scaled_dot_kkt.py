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

# Enflame vendor: dot-free K@K.T - both operands are loaded with the
# same [rows, k] access pattern (no transposed operand) and the tile is
# accumulated as per-k broadcast outer products (no tl.dot, no 2D
# reduction lowering). Correctness-first structure for chips where both
# dot configurations failed.

import torch
import triton
import triton.language as tl

_K_CHUNK = 128


@triton.jit
def _kkt_nodot_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    output_ptr,
    chunk_size,
    k_size,
    nheads,
    ratio,
    k_stride_batch,
    k_stride_seqlen,
    k_stride_head,
    k_stride_k,
    beta_stride_batch,
    beta_stride_seqlen,
    beta_stride_head,
    g_stride_batch,
    g_stride_seqlen,
    g_stride_head,
    output_stride_batch,
    output_stride_seqlen,
    output_stride_head,
    output_stride_last,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_CHUNK: tl.constexpr,
    HAS_G: tl.constexpr,
):
    k_stride_batch = tl.cast(k_stride_batch, tl.int64)
    k_stride_seqlen = tl.cast(k_stride_seqlen, tl.int64)
    k_stride_head = tl.cast(k_stride_head, tl.int64)
    k_stride_k = tl.cast(k_stride_k, tl.int64)
    beta_stride_batch = tl.cast(beta_stride_batch, tl.int64)
    beta_stride_seqlen = tl.cast(beta_stride_seqlen, tl.int64)
    beta_stride_head = tl.cast(beta_stride_head, tl.int64)
    g_stride_batch = tl.cast(g_stride_batch, tl.int64)
    g_stride_seqlen = tl.cast(g_stride_seqlen, tl.int64)
    g_stride_head = tl.cast(g_stride_head, tl.int64)
    output_stride_batch = tl.cast(output_stride_batch, tl.int64)
    output_stride_seqlen = tl.cast(output_stride_seqlen, tl.int64)
    output_stride_head = tl.cast(output_stride_head, tl.int64)
    output_stride_last = tl.cast(output_stride_last, tl.int64)

    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    chunk_head_id = tl.program_id(2)
    chunk_id = chunk_head_id // nheads
    head_id = chunk_head_id - chunk_id * nheads
    group_id = head_id // ratio

    num_n_tiles = tl.cdiv(chunk_size, BLOCK_N)
    m_tile = tile_id // num_n_tiles
    n_tile = tile_id - m_tile * num_n_tiles

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    m_global = chunk_id * chunk_size + m_offsets
    n_global = chunk_id * chunk_size + n_offsets
    m_mask = m_offsets < chunk_size
    n_mask = n_offsets < chunk_size

    k_base = batch_id * k_stride_batch + group_id * k_stride_head
    m_row = k_base + m_global * k_stride_seqlen
    n_row = k_base + n_global * k_stride_seqlen

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k_size, K_CHUNK):
        for kk in tl.static_range(K_CHUNK):
            idx = k_start + kk
            valid = idx < k_size
            a_col = tl.load(
                k_ptr + m_row + idx * k_stride_k,
                mask=m_mask & valid,
                other=0.0,
            ).to(tl.float32)
            b_col = tl.load(
                k_ptr + n_row + idx * k_stride_k,
                mask=n_mask & valid,
                other=0.0,
            ).to(tl.float32)
            accumulator += a_col[:, None] * b_col[None, :]

    beta_base = batch_id * beta_stride_batch + head_id * beta_stride_head
    beta_m = tl.load(
        beta_ptr + beta_base + m_global * beta_stride_seqlen,
        mask=m_mask,
        other=0.0,
    ).to(tl.float32)
    result = accumulator * beta_m[:, None]
    if HAS_G:
        g_base = batch_id * g_stride_batch + head_id * g_stride_head
        g_m = tl.load(
            g_ptr + g_base + m_global * g_stride_seqlen,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        g_n = tl.load(
            g_ptr + g_base + n_global * g_stride_seqlen,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        g_diff = g_m[:, None] - g_n[None, :]
        result = result * tl.where(g_diff <= 0.0, tl.exp(g_diff), 0.0)

    result = tl.where(m_offsets[:, None] > n_offsets[None, :], result, 0.0)
    output_offsets = (
        batch_id * output_stride_batch
        + m_global[:, None] * output_stride_seqlen
        + head_id * output_stride_head
        + n_offsets[None, :] * output_stride_last
    )
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    batch, seqlen, num_k_heads, k_size = k.shape
    num_heads = beta.shape[-1]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")
    if num_heads % num_k_heads:
        raise ValueError("num_heads must be divisible by num_k_heads")
    ratio = num_heads // num_k_heads
    nchunks = seqlen // chunk_size
    output = torch.empty(
        (batch, seqlen, num_heads, chunk_size),
        dtype=torch.float32,
        device=k.device,
    )
    if output.numel() == 0:
        return output

    block_m = 32
    block_n = 32
    if g_cumsum is None:
        g_cumsum = beta
    grid = (
        triton.cdiv(chunk_size, block_m) * triton.cdiv(chunk_size, block_n),
        batch,
        nchunks * num_heads,
    )
    _kkt_nodot_kernel[grid](
        k,
        beta,
        g_cumsum,
        output,
        chunk_size,
        k_size,
        num_heads,
        ratio,
        *k.stride(),
        *beta.stride(),
        *g_cumsum.stride(),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_CHUNK=_K_CHUNK,
        HAS_G=g_cumsum is not beta,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

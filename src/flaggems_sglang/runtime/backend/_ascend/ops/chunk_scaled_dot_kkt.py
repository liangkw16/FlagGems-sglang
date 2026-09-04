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

# Ascend vendor: 64x64 tiles at stages=1 (chunk_state E7 platform-proven
# Ascend Cube optimum, 0.329->2.1185x there); S0 ran 32x32 and scored
# 0.031x, below the 0.1 threshold.


@triton.jit
def _chunk_scaled_dot_kkt_kernel(
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
    BLOCK_K: tl.constexpr,
    USE_INPUT_DTYPE: tl.constexpr,
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
    # GQA: k has Hg = H // ratio heads; beta/g keep the full H heads.
    group_id = head_id // ratio

    num_n_tiles = tl.cdiv(chunk_size, BLOCK_N)
    m_tile = tile_id // num_n_tiles
    n_tile = tile_id - m_tile * num_n_tiles

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    m_global = chunk_id * chunk_size + m_offsets
    n_global = chunk_id * chunk_size + n_offsets

    k_base = batch_id * k_stride_batch + group_id * k_stride_head
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
        current_k = k_block * BLOCK_K + k_offsets
        a = tl.load(
            k_ptr
            + k_base
            + m_global[:, None] * k_stride_seqlen
            + current_k[None, :] * k_stride_k,
            mask=(m_offsets[:, None] < chunk_size) & (current_k[None, :] < k_size),
            other=0.0,
        )
        b = tl.load(
            k_ptr
            + k_base
            + current_k[:, None] * k_stride_k
            + n_global[None, :] * k_stride_seqlen,
            mask=(current_k[:, None] < k_size) & (n_offsets[None, :] < chunk_size),
            other=0.0,
        )
        if not USE_INPUT_DTYPE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, input_precision="ieee")

    # beta scales rows (i index); safe-exp decay scales elementwise with
    # exp(g_i - g_j) only when the exponent is <= 0, else zero.
    beta_base = batch_id * beta_stride_batch + head_id * beta_stride_head
    beta_m = tl.load(
        beta_ptr + beta_base + m_global * beta_stride_seqlen,
        mask=m_offsets < chunk_size,
        other=0.0,
    ).to(tl.float32)
    result = accumulator * beta_m[:, None]
    if HAS_G:
        g_base = batch_id * g_stride_batch + head_id * g_stride_head
        g_m = tl.load(
            g_ptr + g_base + m_global * g_stride_seqlen,
            mask=m_offsets < chunk_size,
            other=0.0,
        ).to(tl.float32)
        g_n = tl.load(
            g_ptr + g_base + n_global * g_stride_seqlen,
            mask=n_offsets < chunk_size,
            other=0.0,
        ).to(tl.float32)
        g_diff = g_m[:, None] - g_n[None, :]
        result = result * tl.where(g_diff <= 0.0, tl.exp(g_diff), 0.0)

    # Strict lower triangular: diagonal and above must be written as
    # exact zeros, so keep the full boundary mask on the store.
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
        mask=(m_offsets[:, None] < chunk_size) & (n_offsets[None, :] < chunk_size),
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

    block_m = 64
    block_n = 64
    grid = (
        triton.cdiv(chunk_size, block_m) * triton.cdiv(chunk_size, block_n),
        batch,
        nchunks * num_heads,
    )
    if g_cumsum is None:
        g_cumsum = beta
    _chunk_scaled_dot_kkt_kernel[grid](
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
        BLOCK_K=64,
        USE_INPUT_DTYPE=k.dtype in (torch.float16, torch.bfloat16),
        HAS_G=g_cumsum is not beta,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

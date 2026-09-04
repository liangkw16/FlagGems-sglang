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

# Ascend vendor (persistent 32-tile, BK=128 single K pass): the 64x64
# whole-tile form requires 2646016 UB bits (over the 1572864 budget);
# this keeps the proven 32x32 output tiles but eliminates the K-loop
# (BK=128) and amortizes launch across batches (persistent batch loop
# inside the kernel). GQA dot sharing retained.

import torch
import triton
import triton.language as tl


@triton.jit
def _kkt_persistent_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    output_ptr,
    batch,
    seqlen,
    chunk_size,
    k_size,
    nheads,
    num_k_heads,
    hpg,
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
    chunk_group_id = tl.program_id(1)
    chunk_id = chunk_group_id // num_k_heads
    i_kg = chunk_group_id - chunk_id * num_k_heads

    num_n_tiles = tl.cdiv(chunk_size, BLOCK_N)
    m_tile = tile_id // num_n_tiles
    n_tile = tile_id - m_tile * num_n_tiles

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    m_mask = m_offsets < chunk_size
    n_mask = n_offsets < chunk_size
    k_mask = k_offsets < k_size

    strict_lower = m_offsets[:, None] > n_offsets[None, :]

    for b in range(0, batch):
        m_global = chunk_id * chunk_size + m_offsets
        n_global = chunk_id * chunk_size + n_offsets
        k_base = b * k_stride_batch + i_kg * k_stride_head

        a = tl.load(
            k_ptr
            + k_base
            + m_global[:, None] * k_stride_seqlen
            + k_offsets[None, :] * k_stride_k,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        bt = tl.load(
            k_ptr
            + k_base
            + k_offsets[:, None] * k_stride_k
            + n_global[None, :] * k_stride_seqlen,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        if not USE_INPUT_DTYPE:
            a = a.to(tl.float32)
            bt = bt.to(tl.float32)
        accumulator = tl.dot(a, bt, input_precision="ieee")

        for i_h_local in range(0, hpg):
            i_h = i_kg * hpg + i_h_local
            beta_base = b * beta_stride_batch + i_h * beta_stride_head
            beta_m = tl.load(
                beta_ptr + beta_base + m_global * beta_stride_seqlen,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            result = accumulator * beta_m[:, None]
            if HAS_G:
                g_base = b * g_stride_batch + i_h * g_stride_head
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
            result = tl.where(strict_lower, result, 0.0)
            output_offsets = (
                b * output_stride_batch
                + m_global[:, None] * output_stride_seqlen
                + i_h * output_stride_head
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
    hpg = num_heads // num_k_heads
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
    block_k = min(triton.next_power_of_2(k_size), 128)
    grid = (
        triton.cdiv(chunk_size, block_m) * triton.cdiv(chunk_size, block_n),
        nchunks * num_k_heads,
    )
    if g_cumsum is None:
        g_cumsum = beta
    _kkt_persistent_kernel[grid](
        k,
        beta,
        g_cumsum,
        output,
        batch,
        seqlen,
        chunk_size,
        k_size,
        num_heads,
        num_k_heads,
        hpg,
        *k.stride(),
        *beta.stride(),
        *g_cumsum.stride(),
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        USE_INPUT_DTYPE=k.dtype in (torch.float16, torch.bfloat16),
        HAS_G=g_cumsum is not beta,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_scaled_dot_kkt"]

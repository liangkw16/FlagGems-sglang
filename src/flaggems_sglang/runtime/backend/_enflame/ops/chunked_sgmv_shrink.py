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
# See the specific language governing permissions and limitations under
# the License.

# Enflame vendor (e9): one persistent launch over a host-built tile
# descriptor table, replacing e7's per-segment host loop of
# index_select -> GEMM launch -> index_copy_. The vendor guide states
# kernel launch overhead is NOT hidden on GCU, so a host loop with one
# launch per segment (hundreds of launches for a batched request mix)
# pays pure scheduling cost, and each launch also oversubscribes the 24
# SIPs with its own grid. Folding every segment into a single launch
# with a grid capped at the SIP count removes both costs at once.
#
# The descriptor table keeps every inner loop shape-static, which is the
# lesson from the T47 e13 grouped-GEMM eval crashes: a fixed-size GEMM
# tile per table entry (num_stages-pipelined K loop with constexpr K
# bound), no data-dependent loop bounds anywhere -- the persistent walk
# is the only loop whose trip count varies. Per-tile cost is 5 scalar
# loads (seg/m0/n0 + seg start + weight index).
#
# The per-tile GEMM itself is byte-for-byte e7's proven configuration
# (BLOCK 64x64, BLOCK_K = next_pow2(K) capped at 128, native-dtype dot
# operands for fp16/bf16 with an fp32 accumulator, ieee only for true
# fp32 inputs, num_warps=4, num_stages=2): single-variable discipline --
# e9 changes only how tiles are scheduled, not how they compute.

import torch
import triton
import triton.language as tl


@triton.jit
def _shrink_tile_persistent_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    permutation_ptr,
    tile_seg_ptr,
    tile_m0_ptr,
    tile_n0_ptr,
    seg_start_ptr,
    seg_widx_ptr,
    seg_len_ptr,
    num_tiles,
    N,
    K,
    x_stride_token,
    x_stride_col,
    w_stride_lora,
    w_stride_out,
    w_stride_k,
    o_stride_token,
    o_stride_col,
    perm_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INPUT_DTYPE: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_col = tl.cast(x_stride_col, tl.int64)
    w_stride_lora = tl.cast(w_stride_lora, tl.int64)
    w_stride_out = tl.cast(w_stride_out, tl.int64)
    w_stride_k = tl.cast(w_stride_k, tl.int64)
    o_stride_token = tl.cast(o_stride_token, tl.int64)
    o_stride_col = tl.cast(o_stride_col, tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for tile in tl.range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        seg = tl.load(tile_seg_ptr + tile)
        m0 = tl.load(tile_m0_ptr + tile)
        n0 = tl.load(tile_n0_ptr + tile)
        seg_start = tl.load(seg_start_ptr + seg)
        seg_len = tl.load(seg_len_ptr + seg)
        w_idx = tl.load(seg_widx_ptr + seg).to(tl.int64)

        m_local = m0 * BLOCK_M + offs_m
        rows_mask = m_local < seg_len
        rows = tl.load(
            permutation_ptr + (seg_start + m_local) * perm_stride,
            mask=rows_mask,
            other=0,
        ).to(tl.int64)

        n = n0 * BLOCK_N + offs_n
        n_mask = n < N

        base_n = w_idx * w_stride_lora + n[None, :].to(tl.int64) * w_stride_out
        base_rows = rows[:, None] * x_stride_token
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            kk = k + offs_k
            k_mask = kk < K
            a = tl.load(
                x_ptr + base_rows + kk[None, :] * x_stride_col,
                mask=rows_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                weights_ptr + base_n + kk[:, None] * w_stride_k,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if not USE_INPUT_DTYPE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")

        tl.store(
            output_ptr
            + rows[:, None] * o_stride_token
            + n[None, :].to(tl.int64) * o_stride_col,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=rows_mask[:, None] & n_mask[None, :],
        )


_BLOCK_M = 64
_BLOCK_N = 64
_GRID_CAP = 24  # GCU300 SIP count; the guide caps GridDim to the hardware


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    output = torch.zeros(S, N, dtype=x.dtype, device=x.device)
    if output.numel() == 0 or batch_info.bs == 0:
        return output

    seg_indptr = batch_info.seg_indptr.detach().cpu().tolist()
    weight_indices = batch_info.weight_indices.detach().cpu().tolist()

    # Host-side descriptor build: one table entry per (segment, M-tile,
    # N-tile) of every active segment. Empty segments and negative
    # weight indices are dropped here -- same skip semantics as the
    # generic kernel, which the platform has accepted on this task.
    starts, lens, widxs = [], [], []
    tile_seg, tile_m0, tile_n0 = [], [], []
    for b in range(batch_info.bs):
        start, end = seg_indptr[b], seg_indptr[b + 1]
        w_idx = weight_indices[b]
        if start == end or w_idx < 0:
            continue
        seg = len(starts)
        starts.append(start)
        lens.append(end - start)
        widxs.append(w_idx)
        for m0 in range(0, end - start, _BLOCK_M):
            for n0 in range(0, N, _BLOCK_N):
                tile_seg.append(seg)
                tile_m0.append(m0 // _BLOCK_M)
                tile_n0.append(n0 // _BLOCK_N)
    num_tiles = len(tile_seg)
    if num_tiles == 0:
        return output

    device = x.device
    tile_seg_t = torch.tensor(tile_seg, dtype=torch.int32, device=device)
    tile_m0_t = torch.tensor(tile_m0, dtype=torch.int32, device=device)
    tile_n0_t = torch.tensor(tile_n0, dtype=torch.int32, device=device)
    seg_start_t = torch.tensor(starts, dtype=torch.int32, device=device)
    seg_len_t = torch.tensor(lens, dtype=torch.int32, device=device)
    seg_widx_t = torch.tensor(widxs, dtype=torch.int32, device=device)

    bk = min(triton.next_power_of_2(max(K, 16)), 128)
    grid = (min(num_tiles, _GRID_CAP),)
    _shrink_tile_persistent_kernel[grid](
        x,
        weights,
        output,
        batch_info.permutation,
        tile_seg_t,
        tile_m0_t,
        tile_n0_t,
        seg_start_t,
        seg_widx_t,
        seg_len_t,
        num_tiles,
        N,
        K,
        *x.stride(),
        *weights.stride(),
        *output.stride(),
        batch_info.permutation.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=bk,
        USE_INPUT_DTYPE=(x.dtype in (torch.float16, torch.bfloat16)),
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["chunked_sgmv_shrink"]

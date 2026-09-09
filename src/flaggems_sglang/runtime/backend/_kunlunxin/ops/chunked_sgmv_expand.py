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

# Kunlunxin vendor, grouped-GEMM form (e13): the same single-launch
# regular-GEMM body as the enflame vendor (this backend's per-segment
# route/materialize loop also capped it near 4x while six strong chips
# run the fused generic at 20-55x).
# the 128x128 tile probe (e12) proved tile geometry is not the cause, so
# the cost is the launch storm plus the host metadata loop. This form
# keeps the compile-proven regular-GEMM body (route/materialize recipe,
# T28 E11 / T37 E4) but launches it ONCE with a capped work loop:
# - wrapper materializes the permuted rows with a single index_select
#   pair (no per-segment gathers, one host sync for max_len at most);
# - the kernel decodes (batch, slice, tile_m, tile_n) from a flat task
#   id with scalar div/mod, reads only scalar segment metadata, and has
#   NO runtime branches and NO indirect operand addresses (GCU/XPU
#   poison per the S0-era compile failures);
# - every load address is clamped into the legal range; tail rows and
#   columns are dropped by the store mask only; the K trip is single by
#   construction (BLOCK_K = next_pow2(rank), the multi-K miscompile
#   workaround) with a rank-constexpr tail mask.

import torch
import triton
import triton.language as tl

_GRID_CAP = 65535


@triton.jit(do_not_specialize=["n_tasks"])
def _sgmv_grouped_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    slice_offsets_ptr,
    n_tasks,
    n_tokens,
    num_lora,
    n_slices,
    tiles_per_slice,
    tiles_m,
    tiles_n,
    a_stride_m,
    a_stride_k,
    w_stride_lora,
    w_stride_out,
    w_stride_k,
    c_stride_m,
    c_stride_n,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_stride_m = tl.cast(a_stride_m, tl.int64)
    a_stride_k = tl.cast(a_stride_k, tl.int64)
    w_stride_lora = tl.cast(w_stride_lora, tl.int64)
    w_stride_out = tl.cast(w_stride_out, tl.int64)
    w_stride_k = tl.cast(w_stride_k, tl.int64)
    c_stride_m = tl.cast(c_stride_m, tl.int64)
    c_stride_n = tl.cast(c_stride_n, tl.int64)

    for task in tl.range(tl.program_id(0), n_tasks, tl.num_programs(0)):
        batch_id = task // (n_slices * tiles_per_slice)
        rem = task - batch_id * (n_slices * tiles_per_slice)
        slice_id = rem // tiles_per_slice
        rem = rem - slice_id * tiles_per_slice
        tile_m = rem // tiles_n
        tile_n = rem - tile_m * tiles_n

        start = tl.load(seg_indptr_ptr + batch_id)
        end = tl.load(seg_indptr_ptr + batch_id + 1)
        seg_len = end - start
        out_start = tl.load(slice_offsets_ptr + slice_id)
        out_end = tl.load(slice_offsets_ptr + slice_id + 1)
        out_size = out_end - out_start

        wid_raw = tl.load(weight_indices_ptr + batch_id)
        wid = tl.maximum(tl.minimum(wid_raw, num_lora - 1), 0)
        rank_w = tl.load(lora_ranks_ptr + wid)

        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Clamped, always-legal addresses; validity is enforced only by
        # the store mask below, so the operand loads carry no mask at
        # all (load-value-dependent masks are the GCU vectorization
        # poison; these are pure address arithmetic).
        row_ok = offs_m < seg_len
        safe_m = tl.maximum(tl.minimum(offs_m, seg_len - 1), 0)
        row_addr = tl.minimum(start + safe_m, n_tokens - 1)
        col_ok = offs_n < out_size
        safe_n = tl.maximum(tl.minimum(offs_n, out_size - 1), 0)
        # an empty trailing slice can start at the very end of the
        # output range; park its (fully masked) column loads at row 0
        col_base = tl.where(out_size > 0, out_start, 0)

        # Multi-trip K with a small fixed BLOCK_K keeps shared memory
        # within budget for large ranks (a single next_pow2(rank) trip
        # needs up to 135KB); addresses are recomputed from k_start each
        # trip, so the historical stride-advance miscompile cannot
        # recur. The K tail is zeroed via the rank-constexpr mask (a
        # clamped duplicate load would wrongly contribute to the dot).
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in tl.range(0, RANK, BLOCK_K):
            k = k_start + offs_k
            k_ok = k < RANK
            safe_k = tl.where(k_ok, k, 0)
            a = tl.load(
                a_ptr
                + row_addr[:, None] * a_stride_m
                + (slice_id * RANK + safe_k[None, :]) * a_stride_k,
                mask=k_ok[None, :],
                other=0.0,
            )
            b = tl.load(
                w_ptr
                + wid * w_stride_lora
                + (col_base + safe_n)[None, :] * w_stride_out
                + safe_k[:, None] * w_stride_k,
                mask=k_ok[:, None],
                other=0.0,
            )
            acc = tl.dot(
                a.to(tl.float32),
                b.to(tl.float32),
                acc,
                input_precision="ieee",
            )

        c_ptrs = (
            c_ptr
            + row_addr[:, None] * c_stride_m
            + (col_base + safe_n)[None, :] * c_stride_n
        )
        mask = (row_ok[:, None] & col_ok[None, :]) & (rank_w != 0)
        base = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)
        scaling = tl.load(scalings_ptr + wid).to(tl.float32)
        tl.store(
            c_ptrs,
            (base + acc * scaling).to(c_ptr.dtype.element_ty),
            mask=mask,
        )


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    output = base_output.clone()
    n_slices = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    num_lora = weights.shape[0]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if (
        output.numel() == 0
        or n_slices <= 0
        or batch_info.bs == 0
        or x.shape[0] == 0
    ):
        return output

    seg_indptr = batch_info.seg_indptr
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return output

    # One full-batch gather pair: permuted x rows and the matching base
    # rows, both contiguous, so the kernel body stays a plain regular
    # GEMM over materialized rows. The platform hands permutation as
    # int32; index ops need long.
    rows = batch_info.permutation.long()
    n_tokens = rows.numel()
    a_mat = x.index_select(0, rows)
    c_mat = output.index_select(0, rows).float()

    block_m = 64
    block_n = 64
    block_k = min(triton.next_power_of_2(max(rank, 16)), 64)
    tiles_m = triton.cdiv(max_len, block_m)
    tiles_n = triton.cdiv(int(max_slice_size), block_n)
    tiles_per_slice = tiles_m * tiles_n
    n_tasks = batch_info.bs * n_slices * tiles_per_slice
    grid = (min(n_tasks, _GRID_CAP),)
    _sgmv_grouped_gemm_kernel[grid](
        a_mat,
        weights,
        c_mat,
        seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        slice_offsets,
        n_tasks,
        n_tokens,
        num_lora,
        n_slices,
        tiles_per_slice,
        tiles_m,
        tiles_n,
        a_mat.stride(0),
        a_mat.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        c_mat.stride(0),
        c_mat.stride(1),
        RANK=rank,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    output.index_copy_(0, rows, c_mat.to(output.dtype))
    return output


__all__ = ["chunked_sgmv_expand"]

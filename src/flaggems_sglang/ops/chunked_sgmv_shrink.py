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

# SGLang-production structure (e1): BLOCK_M = max_len (one program per
# segment, not token-blocked), auto-tuned BLOCK_N/BLOCK_K per shape,
# 2D grid (n_tiles, batch) with segment on axis 1. The S0 3D grid with
# fixed BLOCK_M=64 wasted programs on short segments.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_segs"])
def _sgmv_shrink_sgl_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    num_segs,
    max_out_dim,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    o_stride_0,
    o_stride_1,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    seg_start = tl.load(seg_indptr_ptr + pid_s)
    seg_end = tl.load(seg_indptr_ptr + pid_s + 1)
    if seg_start == seg_end:
        return

    w_index = tl.load(weight_indices_ptr + pid_s)

    pid_n = tl.program_id(0)

    # Map logical sequence index to physical row
    s_logical = tl.arange(0, BLOCK_M) + seg_start
    s_physical = tl.load(permutation_ptr + s_logical, mask=s_logical < seg_end, other=0)

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    # x tile: [BLOCK_M, BLOCK_K]
    x_ptrs = x_ptr + s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    # w tile: [BLOCK_K, BLOCK_N]
    w_ptrs = (
        weights_ptr
        + w_index * w_stride_0
        + (k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1)
    )

    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_logical[:, None] < seg_end) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        partial_sum += tl.dot(x_tile, w_tile, input_precision="ieee")
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    o_ptrs = output_ptr + (
        s_physical[:, None] * o_stride_0 + n_offset[None, :] * o_stride_1
    )
    o_mask = (s_logical[:, None] < seg_end) & (n_offset[None, :] < N)
    tl.store(o_ptrs, partial_sum.to(o_ptrs.dtype.element_ty), mask=o_mask)


def _get_shrink_config(K, N, max_len):
    """Shape-adaptive block sizes (SGLang tuning pattern)."""
    if max_len <= 16:
        return {"BLOCK_N": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    elif max_len <= 32:
        return {"BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4}
    elif max_len <= 64:
        return {"BLOCK_N": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    else:
        return {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3}


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    output = torch.zeros(S, N, dtype=x.dtype, device=x.device)
    if output.numel() == 0 or batch_info.bs == 0:
        return output

    seg_indptr = batch_info.seg_indptr.contiguous()
    weight_indices = batch_info.weight_indices.contiguous()
    permutation = batch_info.permutation.contiguous()

    # BLOCK_M = max segment length (SGLang pattern: one program per segment)
    max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return output

    # Round up to power of 2 for tl.arange
    BLOCK_M = min(max(triton.next_power_of_2(max_len), 16), 64)
    config = _get_shrink_config(K, N, max_len)
    BLOCK_N = min(config["BLOCK_N"], N)
    BLOCK_K = min(config["BLOCK_K"], max(K, 16), 128)  # cap for shared memory
    BLOCK_K = triton.next_power_of_2(max(BLOCK_K, 16))

    grid = (
        triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )
    _sgmv_shrink_sgl_kernel[grid](
        x,
        weights,
        output,
        seg_indptr,
        weight_indices,
        permutation,
        batch_info.bs,
        N,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        K=K,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return output


__all__ = ["chunked_sgmv_shrink"]

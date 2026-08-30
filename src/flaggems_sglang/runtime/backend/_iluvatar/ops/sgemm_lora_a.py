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

_MAX_GRID = 65535
_BLOCK_S = 64
_BLOCK_N = 64
_BLOCK_K = 64


@triton.jit
def _sgemm_lora_a_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    total_tiles,
    input_dim,
    output_dim,
    x_stride_token,
    x_stride_col,
    weight_stride_lora,
    weight_stride_row,
    weight_stride_col,
    output_stride_token,
    output_stride_col,
    seg_indptr_stride,
    weight_indices_stride,
    permutation_stride,
    S_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # flat 1D capped grid over (segment, token-block, output-block) tiles;
    # constexpr block counts keep the div/mod cheap; fp32-ieee dot
    # (fp16-operand dot is numerically unsafe on kunlun/ascend family)
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(pid, total_tiles, grid_stride):
        n_block = tile % N_BLOCKS
        rest = tile // N_BLOCKS
        s_block = rest % S_BLOCKS
        batch_id = rest // S_BLOCKS

        segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
        segment_end = tl.load(
            seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride
        )
        segment_length = segment_end - segment_start
        if s_block * BLOCK_S < segment_length:
            weight_index = tl.load(
                weight_indices_ptr + batch_id * weight_indices_stride
            )
            offsets_s = s_block * BLOCK_S + offs_s
            mask_s = offsets_s < segment_length
            if HAS_PERMUTATION:
                rows = tl.load(
                    permutation_ptr
                    + (segment_start + offsets_s) * permutation_stride,
                    mask=mask_s,
                    other=0,
                )
            else:
                rows = segment_start + offsets_s

            offsets_n = n_block * BLOCK_N + offs_n
            # mask against the ABSOLUTE column index: a per-block arange
            # mask lets n_block>0 tiles write past output_dim when R is
            # not a BLOCK_N multiple (R=65 etc corrupted neighbour rows)
            mask_n = offsets_n < output_dim
            accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, input_dim, BLOCK_K):
                k = k_start + offs_k
                mask_k = k < input_dim
                x = tl.load(
                    x_ptr
                    + rows[:, None] * x_stride_token
                    + k[None, :] * x_stride_col,
                    mask=mask_s[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    weights_ptr
                    + weight_index * weight_stride_lora
                    + offsets_n[None, :] * weight_stride_row
                    + k[:, None] * weight_stride_col,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(tl.float32)
                # tianshu (iluvatar): fp32-operand tl.dot silently
                # mis-executes (T12/T13 platform mirror); split-fp16
                # three-dot carries ~22 mantissa bits - within 1e-4
                x_hi = x.to(tl.float16)
                x_lo = (x - x_hi.to(tl.float32)).to(tl.float16)
                w_hi = w.to(tl.float16)
                w_lo = (w - w_hi.to(tl.float32)).to(tl.float16)
                accumulator += tl.dot(x_hi, w_hi)
                accumulator += tl.dot(x_hi, w_lo)
                accumulator += tl.dot(x_lo, w_hi)
                accumulator += tl.dot(x_lo, w_lo)

            out_mask = mask_s[:, None] & mask_n[None, :]
            out_offsets = (
                rows[:, None] * output_stride_token
                + offsets_n[None, :] * output_stride_col
            )
            tl.store(
                output_ptr + out_offsets,
                accumulator.to(output_ptr.dtype.element_ty),
                mask=out_mask,
            )


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    x = x.contiguous()
    weights = weights.contiguous()
    seq_len, input_dim = x.shape
    output_dim = weights.shape[1]
    output = torch.zeros((seq_len, output_dim), dtype=x.dtype, device=x.device)
    num_segments = int(batch_info.bs)
    if (
        output.numel() == 0
        or num_segments == 0
        or output_dim == 0
        or input_dim == 0
    ):
        return output

    seg_indptr = batch_info.seg_indptr
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        lengths = seg_indptr[1:] - seg_indptr[:-1]
        max_len = int(lengths.max().item()) if lengths.numel() else 0
    if max_len == 0:
        return output

    permutation = batch_info.permutation
    s_blocks = triton.cdiv(max_len, _BLOCK_S)
    n_blocks = triton.cdiv(output_dim, _BLOCK_N)
    total_tiles = num_segments * s_blocks * n_blocks
    grid = (min(total_tiles, _MAX_GRID),)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        seg_indptr,
        batch_info.weight_indices,
        permutation if permutation is not None else seg_indptr,
        total_tiles,
        input_dim,
        output_dim,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        permutation.stride(0) if permutation is not None else 0,
        S_BLOCKS=s_blocks,
        N_BLOCKS=n_blocks,
        HAS_PERMUTATION=permutation is not None,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["sgemm_lora_a"]

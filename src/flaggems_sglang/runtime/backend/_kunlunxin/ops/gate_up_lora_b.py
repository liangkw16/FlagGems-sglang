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

# Kunlunxin vendor v2: host-resolved segment metadata (T13-proven
# pattern). The dot compilation unit contains no dynamic scalar loads
# of seg_indptr/weight_indices/lora_ranks/scalings; permutation gather,
# fp32-ieee dot, tiles and RMW are byte-identical to the generic.

import triton
import triton.language as tl


@triton.jit
def _gate_up_lora_b_kernel(
    x_ptr,
    weights_ptr,
    output_ptr,
    permutation_ptr,
    segment_start,
    segment_length,
    weight_index,
    scaling,
    output_dim,
    matrix_blocks,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
    weight_stride_lora = tl.cast(2 * output_dim * RANK, tl.int64)
    weight_stride_output = tl.cast(RANK, tl.int64)
    weight_stride_rank = tl.cast(1, tl.int64)

    slice_id = tl.program_id(1)
    matrix_pid = tl.program_id(0)

    num_output_blocks = tl.cdiv(output_dim, BLOCK_N)
    token_block = matrix_pid // num_output_blocks
    output_block = matrix_pid - token_block * num_output_blocks

    token_offsets = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < segment_length
    output_mask = output_offsets < output_dim
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr + segment_start + token_offsets,
            mask=token_mask,
            other=0,
        )
    else:
        rows = segment_start + token_offsets

    accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in range(0, RANK, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < RANK
        x_tile = tl.load(
            x_ptr
            + rows[:, None] * (2 * RANK)
            + (slice_id * RANK + k[None, :]),
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            weights_ptr
            + weight_index * weight_stride_lora
            + (slice_id * output_dim + output_offsets[None, :])
            * weight_stride_output
            + k[:, None] * weight_stride_rank,
            mask=k_mask[:, None] & output_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.dot(x_tile, w_tile, input_precision="ieee")

    output_ptrs = (
        output_ptr
        + rows[:, None] * tl.cast(2 * output_dim, tl.int64)
        + (slice_id * output_dim + output_offsets[None, :])
    )
    mask = token_mask[:, None] & output_mask[None, :]
    base = tl.load(output_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptrs, base + accumulator * scaling, mask=mask)


def gate_up_lora_b(x, gate_up_lora_b, batch_info, output_dim, base_output):
    x = x.contiguous()
    gate_up_lora_b = gate_up_lora_b.contiguous()
    output = base_output.contiguous().clone()
    rank = gate_up_lora_b.shape[-1]
    if x.shape[1] != 2 * rank:
        raise ValueError("x width must equal 2 * rank")
    bs = batch_info.bs
    if output.numel() == 0 or bs == 0 or output_dim <= 0 or rank == 0:
        return output

    indptr = batch_info.seg_indptr.tolist()
    weight_indices = batch_info.weight_indices.tolist()
    lora_ranks = batch_info.lora_ranks.tolist()
    scalings = batch_info.scalings.tolist()

    permutation = batch_info.permutation
    block_s = 64
    block_n = 64
    block_k = 64
    matrix_blocks = triton.cdiv(1, block_s) * triton.cdiv(output_dim, block_n)
    max_len = 0
    for b in range(bs):
        start = int(indptr[b])
        length = int(indptr[b + 1]) - start
        if length <= 0:
            continue
        wi = int(weight_indices[b])
        if int(lora_ranks[wi]) == 0:
            continue
        max_len = max(max_len, length)
    if max_len == 0:
        return output
    matrix_blocks = triton.cdiv(max_len, block_s) * triton.cdiv(
        output_dim, block_n
    )

    for b in range(bs):
        start = int(indptr[b])
        length = int(indptr[b + 1]) - start
        if length <= 0:
            continue
        wi = int(weight_indices[b])
        if int(lora_ranks[wi]) == 0:
            continue
        grid = (matrix_blocks, 2)
        _gate_up_lora_b_kernel[grid](
            x,
            gate_up_lora_b,
            output,
            permutation if permutation is not None else batch_info.seg_indptr,
            start,
            length,
            wi,
            float(scalings[wi]),
            output_dim,
            matrix_blocks,
            RANK=rank,
            BLOCK_S=block_s,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            HAS_PERMUTATION=permutation is not None,
            num_warps=4,
            num_stages=2,
        )
    return output


__all__ = ["gate_up_lora_b"]

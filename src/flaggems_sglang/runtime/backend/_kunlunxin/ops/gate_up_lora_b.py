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

# Kunlunxin vendor v3 (e7; e8 re-carrier for the kunlun eval window).
# All five
# prior kunlun attempts (3D grid generic, BLOCK_N 128 generic, 1D fold,
# host-resolved dot v1/v2) hit the same inductor compile-worker crash while
# num_stages=2 and tl.dot stayed constant on this path. This variant removes
# the entire dot lowering surface: explicit fp32 FMA K-loop, num_stages=1 /
# num_warps=4 (the kunlun-proven softcap/moe_sum_reduce convention), int32
# offsets only, host-resolved segment metadata kept from v2. Other seven
# chips keep the generic/vendors unchanged.

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
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
):
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
    for k in tl.range(0, RANK):
        x_col = tl.load(
            x_ptr + rows[:, None] * (2 * RANK) + slice_id * RANK + k,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        w_row = tl.load(
            weights_ptr
            + weight_index * (2 * output_dim * RANK)
            + (slice_id * output_dim + output_offsets[None, :]) * RANK
            + k,
            mask=output_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += x_col * w_row

    output_ptrs = (
        output_ptr
        + rows[:, None] * (2 * output_dim)
        + slice_id * output_dim
        + output_offsets[None, :]
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
    for b in range(bs):
        start = int(indptr[b])
        length = int(indptr[b + 1]) - start
        if length <= 0:
            continue
        wi = int(weight_indices[b])
        if int(lora_ranks[wi]) == 0:
            continue
        matrix_blocks = triton.cdiv(length, block_s) * triton.cdiv(
            output_dim, block_n
        )
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
            RANK=rank,
            BLOCK_S=block_s,
            BLOCK_N=block_n,
            HAS_PERMUTATION=permutation is not None,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["gate_up_lora_b"]

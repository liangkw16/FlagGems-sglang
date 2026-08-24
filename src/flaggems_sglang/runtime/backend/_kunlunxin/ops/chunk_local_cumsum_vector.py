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
def _chunk_local_cumsum_vector_kernel(
    g_ptr,
    output_ptr,
    nheads,
    state_size,
    nchunks,
    total_programs,
    g_stride_b,
    g_stride_t,
    g_stride_h,
    g_stride_s,
    output_stride_b,
    output_stride_t,
    output_stride_h,
    output_stride_s,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_F: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    program_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    feature_blocks = tl.cdiv(nheads * state_size, BLOCK_F)
    for logical_id in range(program_id, total_programs, grid_size):
        pid_batch = logical_id // (nchunks * feature_blocks)
        remainder = logical_id - pid_batch * (nchunks * feature_blocks)
        pid_chunk = remainder // feature_blocks
        pid_feature = remainder - pid_chunk * feature_blocks
        offsets_f = pid_feature * BLOCK_F + tl.arange(0, BLOCK_F)
        offsets_c = tl.arange(0, BLOCK_SIZE)
        offsets_h = offsets_f // state_size
        offsets_s = offsets_f - offsets_h * state_size
        mask_f = offsets_f < nheads * state_size
        mask_c = offsets_c < CHUNK_SIZE
        mask = mask_f[:, None] & mask_c[None, :]
        if REVERSE:
            offsets_t = pid_chunk * CHUNK_SIZE + CHUNK_SIZE - 1 - offsets_c
        else:
            offsets_t = pid_chunk * CHUNK_SIZE + offsets_c

        input_offsets = (
            pid_batch * g_stride_b
            + offsets_t[None, :] * g_stride_t
            + offsets_h[:, None] * g_stride_h
            + offsets_s[:, None] * g_stride_s
        )
        values = tl.load(g_ptr + input_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        output = tl.cumsum(values, axis=1)
        if HAS_SCALE:
            output *= scale
        output_offsets = (
            pid_batch * output_stride_b
            + offsets_t[None, :] * output_stride_t
            + offsets_h[:, None] * output_stride_h
            + offsets_s[:, None] * output_stride_s
        )
        tl.store(output_ptr + output_offsets, output, mask=mask)


def chunk_local_cumsum_vector(g, chunk_size, reverse=False, scale=None):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch, seqlen, nheads, state_size = g.shape
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must contain one value")
        scale = scale.item()
    output = torch.empty(g.shape, dtype=torch.float32, device=g.device)
    if output.numel() == 0:
        return output

    block_size = triton.next_power_of_2(chunk_size)
    block_f = max(1, min(8, 4096 // block_size))
    features = nheads * state_size
    feature_blocks = triton.cdiv(features, block_f)
    total_programs = feature_blocks * (seqlen // chunk_size) * batch
    grid = (min(total_programs, 4096),)
    _chunk_local_cumsum_vector_kernel[grid](
        g,
        output,
        nheads,
        state_size,
        seqlen // chunk_size,
        total_programs,
        g.stride(0),
        g.stride(1),
        g.stride(2),
        g.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        1.0 if scale is None else float(scale),
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
        BLOCK_F=block_f,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=2 if chunk_size <= 8 else 4,
        num_stages=1,
    )
    return output


__all__ = ["chunk_local_cumsum_vector"]

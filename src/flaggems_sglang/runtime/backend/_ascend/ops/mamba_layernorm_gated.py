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
def _mamba_layernorm_gated_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    group_count,
    total_groups,
    stride_x_row,
    stride_x_col,
    stride_weight,
    stride_bias,
    stride_z_row,
    stride_z_col,
    stride_output_row,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    program_id = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for logical_id in range(program_id, total_groups, grid_size):
        row = (logical_id // group_count).to(tl.int64)
        group = (logical_id % group_count).to(tl.int64)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < GROUP_SIZE
        flat_columns = group * GROUP_SIZE + columns

        x = tl.load(
            x_ptr + row * stride_x_row + flat_columns * stride_x_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(
                z_ptr + row * stride_z_row + flat_columns * stride_z_col,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x = x * z * tl.sigmoid(z)

        if IS_RMS_NORM:
            centered = x
        else:
            mean = tl.sum(x, axis=0) / GROUP_SIZE
            centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / GROUP_SIZE
        normalized = centered * tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + flat_columns * stride_weight,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = normalized * weight
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + flat_columns * stride_bias,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            output += bias
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(
                z_ptr + row * stride_z_row + flat_columns * stride_z_col,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            output = output * z * tl.sigmoid(z)

        tl.store(
            output_ptr + row * stride_output_row + flat_columns,
            output,
            mask=mask,
        )


def mamba_layernorm_gated(
    x,
    weight,
    bias,
    eps,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=True,
):
    rows, hidden_size = x.shape
    if group_size is None:
        group_size = hidden_size
    if group_size <= 0 or hidden_size % group_size != 0:
        raise ValueError("group_size must be positive and divide hidden size")
    if weight.numel() != hidden_size:
        raise ValueError("weight must contain one value per hidden element")
    if bias is not None and bias.numel() != hidden_size:
        raise ValueError("bias must contain one value per hidden element")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")

    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output

    weight_row = weight.reshape(-1)
    bias_row = weight_row if bias is None else bias.reshape(-1)
    z_rows = x if z is None else z
    block_size = triton.next_power_of_2(group_size)
    num_warps = min(max(block_size // 256, 4), 8)
    if block_size == 512 and group_size < 512:
        num_warps = 2
    group_count = hidden_size // group_size
    total_groups = rows * group_count
    grid = (min(total_groups, 4096),)

    _mamba_layernorm_gated_kernel[grid](
        x,
        weight_row,
        bias_row,
        z_rows,
        output,
        group_count,
        total_groups,
        x.stride(0),
        x.stride(1),
        weight_row.stride(0),
        bias_row.stride(0),
        z_rows.stride(0),
        z_rows.stride(1),
        output.stride(0),
        eps,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


__all__ = ["mamba_layernorm_gated"]

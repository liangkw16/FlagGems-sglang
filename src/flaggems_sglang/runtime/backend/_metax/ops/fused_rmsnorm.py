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
def _fused_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_x_row,
    stride_x_col,
    stride_weight,
    stride_output_row,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size

    x = tl.load(
        x_ptr + row * stride_x_row + columns * stride_x_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + columns * stride_weight, mask=mask, other=0.0
    ).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / hidden_size
    reciprocal_rms = tl.rsqrt(mean_square + eps)
    output = x * reciprocal_rms * weight

    tl.store(
        output_ptr + row * stride_output_row + columns,
        output,
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output

    hidden_size = x.shape[-1]
    if weight.numel() != hidden_size:
        raise ValueError("weight must contain one value per hidden element")

    x_rows = x.reshape(-1, hidden_size)
    weight_row = weight.reshape(-1)
    output_rows = output.reshape(-1, hidden_size)
    block_size = triton.next_power_of_2(hidden_size)

    _fused_rmsnorm_kernel[(x_rows.shape[0],)](
        x_rows,
        weight_row,
        output_rows,
        x_rows.stride(0),
        x_rows.stride(1),
        weight_row.stride(0),
        output_rows.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
    return output


__all__ = ["fused_rmsnorm"]

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
def _log_scaling_tau_kernel(
    x_ptr,
    tau_ptr,
    out_ptr,
    n_cols,
    x_stride_row,
    o_stride_row,
    tau_stride,
    COL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0) // COL_BLOCKS
    col_block = tl.program_id(0) % COL_BLOCKS
    x_stride_row = tl.cast(x_stride_row, tl.int64)
    o_stride_row = tl.cast(o_stride_row, tl.int64)

    tau = tl.load(tau_ptr + row * tau_stride).to(tl.float32)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    # Full blocks skip the per-lane int compare entirely - the
    # Ascend vector-CMP scalar-degradation fix (T40 E16 pattern);
    # only ragged tails keep the mask.
    if EVEN:
        x = tl.load(x_ptr + row * x_stride_row + offs).to(tl.float32)
        tl.store(
            out_ptr + row * o_stride_row + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
        )
    else:
        mask = offs < n_cols
        x = tl.load(
            x_ptr + row * x_stride_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr + row * o_stride_row + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def log_scaling_tau(x, tau):
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.shape[0]
    if out.numel() == 0:
        return out

    n_cols = x.numel() // rows
    block = min(triton.next_power_of_2(max(n_cols, 1)), 1024)
    col_blocks = triton.cdiv(n_cols, block)
    grid = (rows * col_blocks,)
    _log_scaling_tau_kernel[grid](
        x,
        tau,
        out,
        n_cols,
        x.stride(0),
        out.stride(0),
        tau.stride(0),
        COL_BLOCKS=col_blocks,
        BLOCK=block,
        EVEN=(n_cols % block == 0),
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["log_scaling_tau"]

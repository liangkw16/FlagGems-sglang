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

# Kunlunxin vendor: 2D [BLOCK_ROWS, BLOCK_D] row-block tiles for every
# shape. The generic's per-row form measured 0.578x on this chip (one
# tiny program per row is pure per-program fixed overhead), and the
# e2 multi-row path never fired on the scoring shapes because of its
# dim<=128 && rows>4096 gate. The 2D tile + axis=1 reduction structure
# already passed correctness here in e2; this vendor makes it
# always-on with larger row tiles (element budget 4096 per program)
# and a grid-stride loop capped at 65535 programs (the T45/T48 recipe).

import torch
import triton
import triton.language as tl


@triton.jit
def _l2norm_rowblock_kernel(
    x_ptr,
    out_ptr,
    rows,
    eps,
    row_blocks,
    D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for rb in range(pid, row_blocks, grid_size):
        row = rb * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col = tl.arange(0, BLOCK_D)
        mask = (row[:, None] < rows) & (col[None, :] < D)
        x = tl.load(x_ptr + row[:, None] * D + col[None, :], mask, 0).to(
            tl.float32
        )
        sum_sq = tl.sum(x * x, axis=1)
        rstd = 1.0 / tl.sqrt(sum_sq + eps)
        value = x * rstd[:, None]
        tl.store(out_ptr + row[:, None] * D + col[None, :], value, mask)


def l2norm(x, eps=1e-6):
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // x.shape[-1] if x.numel() else 0
    dim = x.shape[-1]
    if rows * dim == 0:
        return out
    block_d = max(triton.next_power_of_2(dim), 16)
    block_rows = max(1, min(32, 4096 // block_d))
    row_blocks = triton.cdiv(rows, block_rows)
    grid = (min(row_blocks, 65535),)
    _l2norm_rowblock_kernel[grid](
        x,
        out,
        rows,
        float(eps),
        row_blocks,
        D=dim,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["l2norm"]

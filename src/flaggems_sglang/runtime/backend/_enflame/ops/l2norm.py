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

# Enflame vendor: per-chip leaderboard intel shows enflame at 1.21 vs
# the second-best team's 7.31 while the generic kernel (one program per
# row, pinned num_warps=4) serves that chip today. The vendor guide
# states GCU launch overhead is not hidden, so a row-per-program grid of
# thousands of rows oversubscribes the 24 SIPs with tiny programs.
# Walk the rows from a SIP-sized persistent grid instead, carry
# num_stages=3 on tl.range so the loop can pingpong, and leave
# num_warps to the backend default (pinning lost twice on this chip:
# T19-E5, T51-E5).

import torch
import triton
import triton.language as tl

_GRID_CAP = 24  # GCU300 SIP count


@triton.jit
def _l2norm_persistent_kernel(
    x_ptr,
    out_ptr,
    rows,
    dim,
    eps,
    x_stride,
    o_stride,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    for row in tl.range(pid, rows, num_programs, num_stages=3):
        x = tl.load(
            x_ptr + row.to(tl.int64) * x_stride + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        rstd = 1.0 / tl.sqrt(sum_sq + eps)
        out = x * rstd
        tl.store(
            out_ptr + row.to(tl.int64) * o_stride + offs,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def l2norm(x, eps=1e-6):
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // x.shape[-1] if x.numel() else 0
    dim = x.shape[-1]
    if rows * dim == 0:
        return out
    block_d = max(triton.next_power_of_2(dim), 16)
    grid = (min(rows, _GRID_CAP),)
    _l2norm_persistent_kernel[grid](
        x,
        out,
        rows,
        dim,
        eps,
        x.stride(-2) if x.dim() > 1 else 1,
        out.stride(-2) if out.dim() > 1 else 1,
        BLOCK_D=block_d,
    )
    return out


__all__ = ["l2norm"]

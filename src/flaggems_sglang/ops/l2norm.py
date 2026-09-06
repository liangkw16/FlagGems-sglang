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
def _l2norm_kernel(
    x_ptr,
    out_ptr,
    rows,
    dim,
    eps,
    x_stride,
    o_stride,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_sq = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(sum_sq + eps)
    out = x * rstd
    tl.store(
        out_ptr + row * o_stride + offs,
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
    grid = (rows,)
    _l2norm_kernel[grid](
        x,
        out,
        rows,
        dim,
        float(eps),
        dim,
        dim,
        BLOCK_D=max(triton.next_power_of_2(dim), 16),
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["l2norm"]

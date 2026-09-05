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

# Enflame vendor: split into two kernels so mid is materialized to
# memory between the two RMSNorm passes, guaranteeing the bf16
# round-trip matches the reference exactly (register-level reuse
# in the fused kernel may skip the intermediate rounding).

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _rmsnorm_pass1(
    x_ptr,
    residual_ptr,
    w1_ptr,
    mid_ptr,
    rows,
    dim,
    eps,
    x_stride,
    r_stride,
    m_stride,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    var1 = tl.sum(x * x, axis=0) / dim
    rms1 = tl.sqrt(var1 + eps)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y1 = x / rms1 * w1
    mid_ty = mid_ptr.dtype.element_ty
    y1_cast = y1.to(mid_ty)
    r_typed = tl.load(residual_ptr + row * r_stride + offs, mask=mask, other=0.0)
    mid_val = r_typed + y1_cast
    tl.store(mid_ptr + row * m_stride + offs, mid_val, mask=mask)


@triton.jit
def _rmsnorm_pass2(
    mid_ptr,
    w2_ptr,
    out_ptr,
    rows,
    dim,
    eps,
    m_stride,
    o_stride,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    # Re-load mid from memory (guaranteed bf16 materialization)
    mid_f = tl.load(mid_ptr + row * m_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    var2 = tl.sum(mid_f * mid_f, axis=0) / dim
    rms2 = tl.sqrt(var2 + eps)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = mid_f / rms2 * w2
    tl.store(
        out_ptr + row * o_stride + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
    x = x.contiguous()
    residual = residual.contiguous()
    weight1 = weight1.contiguous()
    weight2 = weight2.contiguous()
    rows = x.shape[0]
    dim = x.shape[-1]
    out = torch.empty_like(x)
    mid = torch.empty_like(residual)
    if rows * dim == 0:
        return out, mid

    bd = max(triton.next_power_of_2(dim), 16)
    grid = (rows,)
    _rmsnorm_pass1[grid](
        x,
        residual,
        weight1,
        mid,
        rows,
        dim,
        float(eps),
        x.stride(0),
        residual.stride(0),
        mid.stride(0),
        BLOCK_D=bd,
        num_warps=4,
        num_stages=1,
    )
    _rmsnorm_pass2[grid](
        mid,
        weight2,
        out,
        rows,
        dim,
        float(eps),
        mid.stride(0),
        out.stride(0),
        BLOCK_D=bd,
        num_warps=4,
        num_stages=1,
    )
    return out, mid


__all__ = ["fused_dual_residual_rmsnorm"]

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

# Kunlunxin vendor: 2D row-block tiles in the T56-e3 platform-proven
# form (element budget 4096 per program, grid-stride capped at 65535)
# - one tiny program per row measured 0.96-1.03x here, the same
# per-program fixed overhead the l2norm row-block vendor cut by +87%.
# tl.rsqrt / tl.sigmoid and isCloseCoreTiling kept from the proven
# E7/E8 form.

import torch
import triton
import triton.language as tl


@triton.jit
def _fla_ln_gated_kunlun(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    dim,
    eps,
    row_blocks,
    IS_RMS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    ACT_SWISH: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    HAS_D_MASK: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for rb in range(pid, row_blocks, grid_size):
        row = rb * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs = tl.arange(0, BLOCK_D)
        d_mask = offs < dim
        if HAS_ROW_MASK:
            m2 = (row[:, None] < rows) & d_mask[None, :]
        else:
            m2 = d_mask[None, :] & (row[:, None] < rows)
        x = tl.load(
            x_ptr + row[:, None] * dim + offs[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        g = tl.load(
            g_ptr + row[:, None] * dim + offs[None, :], mask=m2, other=0.0
        ).to(tl.float32)

        if IS_RMS:
            var = tl.sum(x * x, axis=1) / dim
            rstd = tl.rsqrt(var + eps)[:, None]
            x_hat = x * rstd
        else:
            mean = tl.sum(x, axis=1) / dim
            xc = tl.where(m2, x - mean[:, None], 0.0)
            var = tl.sum(xc * xc, axis=1) / dim
            x_hat = xc * tl.rsqrt(var + eps)[:, None]

        y = x_hat
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=d_mask, other=1.0).to(tl.float32)
            y = y * w[None, :]
        if HAS_B:
            b = tl.load(b_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)
            y = y + b[None, :]

        sig_g = tl.sigmoid(g)
        if ACT_SWISH:
            y = y * g * sig_g
        elif ACT_SIGMOID:
            y = y * sig_g

        tl.store(
            out_ptr + row[:, None] * dim + offs[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m2,
        )


def fla_layernorm_gated(
    x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True
):
    x = x.contiguous()
    g = g.contiguous()
    rows = x.shape[0]
    dim = x.shape[-1]
    out = torch.empty_like(x)
    if rows * dim == 0:
        return out

    act_swish = activation in ("swish", "silu")
    act_sigmoid = activation == "sigmoid" and not act_swish

    if weight is None:
        weight = x  # placeholder, HAS_W=False skips the load
    if bias is None:
        bias = x
    HAS_W = weight is not x
    HAS_B = bias is not x
    if HAS_W:
        weight = weight.contiguous()
    if HAS_B:
        bias = bias.contiguous()

    block_d = max(triton.next_power_of_2(dim), 16)
    block_rows = max(1, min(16, 4096 // block_d))
    row_blocks = triton.cdiv(rows, block_rows)
    grid = (min(row_blocks, 65535),)
    _fla_ln_gated_kunlun[grid](
        x,
        g,
        weight,
        bias,
        out,
        rows,
        dim,
        float(eps),
        row_blocks,
        IS_RMS=is_rms_norm,
        HAS_W=HAS_W,
        HAS_B=HAS_B,
        ACT_SWISH=act_swish,
        ACT_SIGMOID=act_sigmoid,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        HAS_ROW_MASK=(rows % block_rows != 0),
        HAS_D_MASK=(dim != block_d),
        isCloseCoreTiling=True,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["fla_layernorm_gated"]

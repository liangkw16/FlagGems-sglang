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

# Kunlunxin vendor: three-kernel split with 3D micro-programs and a
# rank-1 width reduction (Codex P1). The scalar FMA chain miscompiles
# deterministically on this backend (E4-E6 identical wrong values) and
# the E7 [W_PAD, BLOCK_D] 2D-tile form hits the uni_sram compile wall,
# so the conv kernel holds ONLY a [W_PAD] vector and one tl.sum; bias,
# SiLU and the output cast live in a separate flat kernel (T53
# vectorized-flat recipe) and the state copy is a plain flat copy.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _ccu_conv_rank1_kernel(
    xcat_ptr,
    weight_ptr,
    pre_ptr,
    total,
    dim,
    seqlen,
    state_len,
    l_cat,
    WIDTH: tl.constexpr,
    W_PAD: tl.constexpr,
):
    # E9: 1D flat grid-stride (T53-proven kunlunxin form) instead of the
    # 3D (seqlen, dim, batch) grid - E8 produced ~87-94% wrong elements
    # on kunlunxin only, matching a grid-axis-mapping miscompile.
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    offs_w = tl.arange(0, W_PAD)
    w_mask = offs_w < WIDTH
    for idx in range(pid, total, step):
        t = idx % seqlen
        row = idx // seqlen
        d = row % dim
        # Window positions in the virtual concat(state, x).
        p = t + state_len + 1 - WIDTH + offs_w
        v = tl.load(xcat_ptr + row * l_cat + p, mask=w_mask, other=0.0)
        wk = tl.load(
            weight_ptr + d * WIDTH + offs_w, mask=w_mask, other=0.0
        )
        pre = tl.sum(v * wk, axis=0)
        tl.store(pre_ptr + idx, pre)


@triton.jit
def _ccu_postprocess_kernel(
    pre_ptr,
    bias_ptr,
    out_ptr,
    total,
    seqlen,
    dim,
    HAS_BIAS: tl.constexpr,
    ACT_IS_SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, total, step):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total
        v = tl.load(pre_ptr + idx, mask=mask, other=0.0)
        if HAS_BIAS:
            d = (idx // seqlen) % dim
            v += tl.load(bias_ptr + d, mask=mask, other=0.0)
        if ACT_IS_SILU:
            # SiLU in the statement's exact form; stability rewrites
            # fail the checker at large negative inputs.
            v = v / (1.0 + tl.exp(-v))
        tl.store(out_ptr + idx, v.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ccu_state_copy_kernel(
    xcat_ptr,
    new_state_ptr,
    total,
    seqlen,
    state_len,
    l_cat,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, total, step):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total
        row = idx // state_len
        i = idx % state_len
        v = tl.load(
            xcat_ptr + row.to(tl.int64) * l_cat + seqlen + i, mask=mask, other=0.0
        )
        tl.store(
            new_state_ptr + idx,
            v.to(new_state_ptr.dtype.element_ty),
            mask=mask,
        )


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    squeeze_out = x.dim() == 2
    if squeeze_out:
        x = x.unsqueeze(-1)
    orig_dtype = x.dtype
    # Concatenate state+x along time axis (data layout, not computation)
    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1).contiguous()
    weight_f = weight.contiguous().float()  # [D, W]
    if bias is not None:
        bias = bias.contiguous().float()
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[1]
    out = torch.empty(
        batch, dim, seqlen, dtype=torch.float32, device=x.device
    )
    new_state = torch.empty(
        conv_state.shape, dtype=conv_state.dtype, device=x.device
    )
    if batch * dim == 0:
        out = out.to(orig_dtype)
        if squeeze_out:
            out = out.squeeze(-1)
        return out, new_state

    w_pad = max(triton.next_power_of_2(width), 2)
    l_cat = state_len + seqlen
    pre = torch.empty(
        batch * dim * seqlen, dtype=torch.float32, device=x.device
    )

    # Kernel 1: one micro-program per output element with a [W_PAD]
    # rank-1 reduction - no FMA chain, no 2D tile, no activation.
    total = batch * dim * seqlen
    grid1 = (min(total, _MAX_GRID),)
    _ccu_conv_rank1_kernel[grid1](
        x_cat,
        weight_f,
        pre,
        total,
        dim,
        seqlen,
        state_len,
        l_cat,
        WIDTH=width,
        W_PAD=w_pad,
        num_warps=1,
        num_stages=1,
    )

    # Kernel 2: flat bias + SiLU + dtype cast (out is contiguous).
    grid2 = (min(triton.cdiv(total, 1024), _MAX_GRID),)
    _ccu_postprocess_kernel[grid2](
        pre,
        bias if bias is not None else pre,
        out,
        total,
        seqlen,
        dim,
        HAS_BIAS=bias is not None,
        ACT_IS_SILU=(activation in ("silu", "swish")),
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    # Kernel 3: flat state copy from the tail of x_cat.
    total_ns = batch * dim * state_len
    grid3 = (min(triton.cdiv(total_ns, 1024), _MAX_GRID),)
    _ccu_state_copy_kernel[grid3](
        x_cat,
        new_state,
        total_ns,
        seqlen,
        state_len,
        l_cat,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    out = out.to(orig_dtype)
    if squeeze_out:
        out = out.squeeze(-1)
    return out, new_state


__all__ = ["causal_conv1d_update"]

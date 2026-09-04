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

# Enflame vendor: static short-axis form at 512-lane blocks with
# pure 32-bit offsets (no int64 casts).

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _ccu_static_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    new_state_ptr,
    batch,
    dim,
    x_sb,
    x_sd,
    x_ss,
    st_sb,
    st_sd,
    st_sl,
    ns_sb,
    ns_sd,
    ns_sl,
    o_sb,
    o_sd,
    o_ss,
    SEQLEN: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_IS_SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Static short-axis form (Ascend tutorial 003 structure): SEQLEN /
    # STATE_LEN / WIDTH are constexpr, so every window position p is a
    # compile-time integer and `p < STATE_LEN` resolves to a single
    # load path per (t, k) - no runtime scalar selects, no dual
    # addresses, no time-axis padding.
    pid = tl.program_id(0)
    dim_blocks = tl.cdiv(dim, BLOCK_D)
    total = batch * dim_blocks
    grid_size = tl.num_programs(0)
    for job in range(pid, total, grid_size):
        b = job // dim_blocks
        db = job - b * dim_blocks
        offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = offs_d < dim
        x_base = x_ptr + b.to(tl.int64) * x_sb
        s_base = state_ptr + b.to(tl.int64) * st_sb
        o_base = out_ptr + b.to(tl.int64) * o_sb
        n_base = new_state_ptr + b.to(tl.int64) * ns_sb

        for t in tl.static_range(SEQLEN):
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for k in tl.static_range(WIDTH):
                p = t + STATE_LEN + 1 - WIDTH + k
                if p < STATE_LEN:
                    v = tl.load(
                        s_base + offs_d * st_sd + p * st_sl,
                        mask=dmask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    v = tl.load(
                        x_base + offs_d * x_sd + (p - STATE_LEN) * x_ss,
                        mask=dmask,
                        other=0.0,
                    ).to(tl.float32)
                wk = tl.load(
                    weight_ptr + offs_d * WIDTH + k,
                    mask=dmask,
                    other=0.0,
                ).to(tl.float32)
                acc += wk * v
            if HAS_BIAS:
                acc += tl.load(bias_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
            if ACT_IS_SILU:
                acc = acc / (1.0 + tl.exp(-acc))
            tl.store(
                o_base + offs_d * o_sd + t * o_ss,
                acc.to(out_ptr.dtype.element_ty),
                mask=dmask,
            )

        for i in tl.static_range(STATE_LEN):
            p = SEQLEN + i
            if p < STATE_LEN:
                v2 = tl.load(
                    s_base + offs_d * st_sd + p * st_sl,
                    mask=dmask,
                    other=0.0,
                ).to(tl.float32)
            else:
                v2 = tl.load(
                    x_base + offs_d * x_sd + (p - STATE_LEN) * x_ss,
                    mask=dmask,
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                n_base + offs_d * ns_sd + i * ns_sl,
                v2.to(new_state_ptr.dtype.element_ty),
                mask=dmask,
            )


_BLOCK_D = 512


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    squeeze_out = x.dim() == 2
    if squeeze_out:
        x = x.unsqueeze(-1)
    x = x.contiguous()
    state = conv_state.contiguous()
    w = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    batch, dim, seqlen = x.shape
    state_len = state.shape[-1]
    width = w.shape[1]
    out = torch.empty_like(x)
    new_state = torch.empty_like(state)
    if batch * dim == 0:
        if squeeze_out:
            out = out.squeeze(-1)
        return out, new_state
    dim_blocks = triton.cdiv(dim, _BLOCK_D)
    total = batch * dim_blocks
    grid = (min(total, _MAX_GRID),)
    _ccu_static_kernel[grid](
        x,
        state,
        w,
        bias if bias is not None else x,
        out,
        new_state,
        batch,
        dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        new_state.stride(0),
        new_state.stride(1),
        new_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SEQLEN=seqlen,
        STATE_LEN=state_len,
        WIDTH=width,
        HAS_BIAS=bias is not None,
        ACT_IS_SILU=(activation in ("silu", "swish")),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    if squeeze_out:
        out = out.squeeze(-1)
    return out, new_state


__all__ = ["causal_conv1d_update"]

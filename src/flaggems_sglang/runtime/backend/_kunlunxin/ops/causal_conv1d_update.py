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

_BLOCK_D = 64

# Kunlunxin vendor: BLOCK_D=64 lanes (T36 E27 64-core alignment) plus
# the isCloseCoreTiling launch hint (T36 E13); S0 failed correctness.


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    new_state_ptr,
    batch,
    dim,
    seqlen,
    state_len,
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
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_IS_SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
):
    # One program owns a (batch, dim-block) strip and walks the small
    # seqlen/state axes as scalar loops, so every load/store stays 1D
    # (the T36 kunlunxin poison is 2D masked tiles + transcendental +
    # axis-1 reduce in one compilation unit).
    db = tl.program_id(0)
    b = tl.program_id(1)
    offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = offs_d < dim
    offs_d64 = offs_d.to(tl.int64)
    x_base = x_ptr + b.to(tl.int64) * x_sb
    s_base = state_ptr + b.to(tl.int64) * st_sb
    o_base = out_ptr + b.to(tl.int64) * o_sb
    n_base = new_state_ptr + b.to(tl.int64) * ns_sb

    for t in range(0, seqlen):
        val = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in tl.static_range(WIDTH):
            # Window position in the virtual concat(state, x).
            p = t + state_len + 1 - WIDTH + k
            from_state = p < state_len
            safe_s = tl.minimum(tl.maximum(p, 0), state_len - 1)
            safe_x = tl.minimum(tl.maximum(p - state_len, 0), seqlen - 1)
            s_v = tl.load(
                s_base + offs_d64 * st_sd + safe_s * st_sl,
                mask=dmask & from_state,
                other=0.0,
            ).to(tl.float32)
            x_v = tl.load(
                x_base + offs_d64 * x_sd + safe_x * x_ss,
                mask=dmask & (p >= state_len),
                other=0.0,
            ).to(tl.float32)
            v = tl.where(from_state, s_v, x_v)
            wk = tl.load(
                weight_ptr + offs_d64 * WIDTH + k,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            val += wk * v
        if HAS_BIAS:
            val += tl.load(bias_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
        if ACT_IS_SILU:
            # SiLU in the statement's exact form; stability rewrites
            # fail the checker at large negative inputs.
            val = val / (1.0 + tl.exp(-val))
        tl.store(
            o_base + offs_d64 * o_sd + t * o_ss,
            val.to(out_ptr.dtype.element_ty),
            mask=dmask,
        )

    for i in range(0, state_len):
        p = seqlen + i
        from_state = p < state_len
        safe_s = tl.minimum(tl.maximum(p, 0), state_len - 1)
        safe_x = tl.minimum(tl.maximum(p - state_len, 0), seqlen - 1)
        s_v = tl.load(
            s_base + offs_d64 * st_sd + safe_s * st_sl,
            mask=dmask & from_state,
            other=0.0,
        ).to(tl.float32)
        x_v = tl.load(
            x_base + offs_d64 * x_sd + safe_x * x_ss,
            mask=dmask & (p >= state_len),
            other=0.0,
        ).to(tl.float32)
        v = tl.where(from_state, s_v, x_v)
        tl.store(
            n_base + offs_d64 * ns_sd + i * ns_sl,
            v.to(new_state_ptr.dtype.element_ty),
            mask=dmask,
        )


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
    grid = (triton.cdiv(dim, _BLOCK_D), batch)
    _causal_conv1d_update_kernel[grid](
        x,
        state,
        w,
        bias if bias is not None else x,
        out,
        new_state,
        batch,
        dim,
        seqlen,
        state_len,
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
        WIDTH=width,
        HAS_BIAS=bias is not None,
        ACT_IS_SILU=(activation in ("silu", "swish")),
        BLOCK_D=_BLOCK_D,
        isCloseCoreTiling=True,
    )
    if squeeze_out:
        out = out.squeeze(-1)
    return out, new_state


__all__ = ["causal_conv1d_update"]

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

# Ascend vendor: width-axis reduction form. Wrapper concatenates
# state+x into one buffer and pre-transposes weight to [W, D]; the
# kernel loads a [W_PAD, D_BLOCK] tile per output step and computes
# tl.sum(w * v, axis=0) - replacing the scalar FMA chain that
# miscompiles on this backend. No time-axis padding.

import torch
import triton
import triton.language as tl

_BLOCK_D = 256
_MAX_GRID = 65535


@triton.jit
def _ccu_width_reduce_kernel(
    xcat_ptr,
    wt_ptr,
    bias_ptr,
    out_ptr,
    new_state_ptr,
    batch,
    dim,
    seqlen,
    state_len,
    xcat_sb,
    xcat_sd,
    xcat_sp,
    wt_stride_w,
    wt_stride_d,
    ns_sb,
    ns_sd,
    ns_sl,
    o_sb,
    o_sd,
    o_ss,
    SEQLEN: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    W_PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_IS_SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_blocks = tl.cdiv(dim, BLOCK_D)
    total = batch * dim_blocks
    grid_size = tl.num_programs(0)

    offs_w = tl.arange(0, W_PAD)
    w_mask = offs_w < WIDTH
    offs_d_base = tl.arange(0, BLOCK_D)

    for job in range(pid, total, grid_size):
        b = job // dim_blocks
        db = job - b * dim_blocks
        offs_d = db * BLOCK_D + offs_d_base
        dmask = offs_d < dim

        xcat_row = xcat_ptr + b * xcat_sb + offs_d * xcat_sd
        wt_row = wt_ptr + offs_d * wt_stride_d

        for t in tl.static_range(SEQLEN):
            # Window start: the first of `width` consecutive positions
            # ending at the new token t
            win_start = t + state_len + 1 - WIDTH
            p = win_start + offs_w  # [W_PAD] positions in x_cat
            # 2D tile [W_PAD, BLOCK_D]
            window = tl.load(
                xcat_row + p[:, None] * xcat_sp,
                mask=w_mask[:, None] & dmask[None, :],
                other=0.0,
            ).to(tl.float32)
            wk = tl.load(
                wt_row + offs_w[:, None] * wt_stride_w,
                mask=w_mask[:, None] & dmask[None, :],
                other=0.0,
            ).to(tl.float32)
            # Width-axis reduction: [W_PAD, BLOCK_D] -> [BLOCK_D]
            acc = tl.sum(window * wk, axis=0)

            if HAS_BIAS:
                acc += tl.load(bias_ptr + offs_d, mask=dmask, other=0.0).to(tl.float32)
            if ACT_IS_SILU:
                acc = acc * tl.sigmoid(acc)
            tl.store(
                out_ptr + b * o_sb + offs_d * o_sd + t * o_ss,
                acc.to(out_ptr.dtype.element_ty),
                mask=dmask,
            )

        # New state: copy last STATE_LEN positions from x_cat
        state_offs = tl.arange(0, 64)
        for i0 in range(0, state_len, 64):
            si = i0 + state_offs
            smask = si < state_len
            src_p = seqlen + si
            xcat_b2 = xcat_ptr + b * xcat_sb + offs_d[None, :] * xcat_sd
            v = tl.load(
                xcat_b2 + src_p[:, None] * xcat_sp,
                mask=smask[:, None] & dmask[None, :],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                new_state_ptr
                + b * ns_sb
                + offs_d[None, :] * ns_sd
                + si[:, None] * ns_sl,
                v.to(new_state_ptr.dtype.element_ty),
                mask=smask[:, None] & dmask[None, :],
            )


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    squeeze_out = x.dim() == 2
    if squeeze_out:
        x = x.unsqueeze(-1)
    orig_dtype = x.dtype
    # Concatenate state+x along time axis (data layout, not computation)
    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1).contiguous()
    wt = weight.t().contiguous().float()  # [W, D]
    if bias is not None:
        bias = bias.contiguous().float()
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[1]
    out = torch.empty(batch, dim, seqlen, dtype=torch.float32, device=x.device)
    new_state = torch.empty_like(conv_state)
    if batch * dim == 0:
        out = out.to(orig_dtype)
        if squeeze_out:
            out = out.squeeze(-1)
        return out, new_state

    w_pad = max(triton.next_power_of_2(width), 2)
    dim_blocks = triton.cdiv(dim, _BLOCK_D)
    total = batch * dim_blocks
    grid = (min(total, _MAX_GRID),)
    _ccu_width_reduce_kernel[grid](
        x_cat,
        wt,
        bias if bias is not None else x_cat,
        out,
        new_state,
        batch,
        dim,
        seqlen,
        state_len,
        x_cat.stride(0),
        x_cat.stride(1),
        x_cat.stride(2),
        wt.stride(0),
        wt.stride(1),
        new_state.stride(0),
        new_state.stride(1),
        new_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SEQLEN=seqlen,
        STATE_LEN=state_len,
        WIDTH=width,
        W_PAD=w_pad,
        HAS_BIAS=bias is not None,
        ACT_IS_SILU=(activation in ("silu", "swish")),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    out = out.to(orig_dtype)
    if squeeze_out:
        out = out.squeeze(-1)
    return out, new_state


__all__ = ["causal_conv1d_update"]

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

# Channel-contiguous affine loads adapted from FlagGems-sglang PR #34,
# e7f91a5f6c813d499275b3f3a6288e1b3b5dddc9, causal_conv1d_fn.
# Materialization changes layout only; convolution and SiLU stay in Triton.
import torch
import triton
import triton.language as tl


@triton.jit
def _ccu_affine_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    DIM: tl.constexpr,
    SEQLEN: tl.constexpr,
    STATE: tl.constexpr,
    WIDTH: tl.constexpr,
    D_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // D_BLOCKS
    d = (pid % D_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
    b = row // SEQLEN
    t = row % SEQLEN
    mask = d < DIM
    base = (b * (STATE + SEQLEN) + t + STATE + 1 - WIDTH) * DIM
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(WIDTH):
        xv = tl.load(x_ptr + base + k * DIM + d, mask=mask, other=0.0).to(
            tl.float32
        )
        wv = tl.load(w_ptr + k * DIM + d, mask=mask, other=0.0).to(tl.float32)
        acc = acc + xv * wv
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + d, mask=mask, other=0.0).to(tl.float32)
    if HAS_ACT:
        acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + row * DIM + d, acc, mask=mask)


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
            xcat_ptr + row.to(tl.int64) * l_cat + seqlen + i,
            mask=mask,
            other=0.0,
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
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[1]
    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1)
    x_t = x_cat.permute(0, 2, 1).contiguous()
    w_t = weight.t().contiguous()
    out_t = torch.empty((batch, seqlen, dim), dtype=x.dtype, device=x.device)
    new_state = torch.empty(
        conv_state.shape, dtype=conv_state.dtype, device=x.device
    )
    if batch * dim * seqlen:
        block = min(triton.next_power_of_2(dim), 1024)
        d_blocks = triton.cdiv(dim, block)
        _ccu_affine_kernel[(batch * seqlen * d_blocks,)](
            x_t,
            w_t,
            bias.contiguous() if bias is not None else w_t,
            out_t,
            DIM=dim,
            SEQLEN=seqlen,
            STATE=state_len,
            WIDTH=width,
            D_BLOCKS=d_blocks,
            HAS_BIAS=bias is not None,
            HAS_ACT=activation in ("silu", "swish"),
            BLOCK=block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    total = batch * dim * state_len
    if total:
        _ccu_state_copy_kernel[(min(triton.cdiv(total, 1024), 65535),)](
            x_cat,
            new_state,
            total,
            seqlen,
            state_len,
            state_len + seqlen,
            BLOCK=1024,
            num_warps=4,
            num_stages=1,
        )
    out = out_t.permute(0, 2, 1).contiguous()
    return (out.squeeze(-1) if squeeze_out else out), new_state


__all__ = ["causal_conv1d_update"]

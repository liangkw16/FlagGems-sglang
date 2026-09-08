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
#
# E16 byte diet: the pack buffer is built straight in the
# channel-contiguous [B, L+S, D] layout in the input dtype (one aten
# transpose-copy instead of the fp32 cast + cat + permute chain; the
# kernel's .to(tl.float32) loads are the exact cast the reference
# applies first), and new_state is the tail slice of that buffer
# viewed back to [B, D, L] instead of a second Triton copy kernel.

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


def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    squeeze_out = x.dim() == 2
    if squeeze_out:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    width = weight.shape[1]
    # Pack state+x directly into the channel-contiguous [B, L+S, D]
    # layout in the input dtype (mixed dtypes promote like the
    # reference's leading .float() pair).
    x_t = torch.cat((conv_state.permute(0, 2, 1), x.permute(0, 2, 1)), dim=1)
    w_t = weight.t().contiguous()
    out_t = torch.empty((batch, seqlen, dim), dtype=x.dtype, device=x.device)
    if batch * dim * seqlen:
        # E18 platform run: 256-wide blocks lost 72% on this chip - the
        # per-program vector width dominates, so keep the proven 1024 cap
        # and only grow it if the grid.x limit would be exceeded.
        block = min(triton.next_power_of_2(dim), 1024)
        while batch * seqlen * triton.cdiv(dim, block) > 65535:
            block *= 2
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
    # new_state is the pack buffer's tail slice viewed back to
    # [B, D, L]; same dtype is a zero-copy view, a promoted dtype
    # materializes the cast copy.
    new_state = x_t[:, seqlen:, :].permute(0, 2, 1).to(conv_state.dtype)
    out = out_t.permute(0, 2, 1).contiguous()
    return (out.squeeze(-1) if squeeze_out else out), new_state


__all__ = ["causal_conv1d_update"]

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

# metax vendor: the hygon 2D path-split form measured +31% on
# Haiguang (fused_eh_norm e1, 10.92 -> 14.26) and the per-chip
# leaderboard leaves this chip +0.75 (muxi 6.21 vs the leader's
# 6.96), so it gets the same shape: one program per (token, path),
# halving per-program serial work. Generic and the hygon vendor
# stay byte-identical.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_eh_norm_split(
    embeds_ptr,
    prev_ptr,
    enorm_ptr,
    hnorm_ptr,
    out_ptr,
    es0,
    es1,
    ps0,
    ps1,
    ws0,
    hs0,
    os0,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < hidden
    if which == 0:
        x = tl.load(
            embeds_ptr + row * es0 + cols * es1, mask=mask, other=0.0
        ).to(tl.float32)
        w = tl.load(enorm_ptr + cols * ws0, mask=mask, other=0.0).to(
            tl.float32
        )
        ms = tl.sum(x * x, axis=0) / hidden
        y = x * tl.rsqrt(ms + eps) * w
        tl.store(
            out_ptr + row * os0 + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        x = tl.load(
            prev_ptr + row * ps0 + cols * ps1, mask=mask, other=0.0
        ).to(tl.float32)
        w = tl.load(hnorm_ptr + cols * hs0, mask=mask, other=0.0).to(
            tl.float32
        )
        ms = tl.sum(x * x, axis=0) / hidden
        y = x * tl.rsqrt(ms + eps) * w
        tl.store(
            out_ptr + row * os0 + hidden + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def fused_eh_norm(
    inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
):
    assert inputs_embeds.ndim == 2
    tokens, hidden = inputs_embeds.shape
    assert previous_hidden.shape == (tokens, hidden)
    assert enorm_weight.numel() == hidden
    assert hnorm_weight.numel() == hidden
    out = torch.empty(
        (tokens, hidden * 2),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if tokens and hidden:
        _fused_eh_norm_split[(tokens, 2)](
            inputs_embeds,
            previous_hidden,
            enorm_weight,
            hnorm_weight,
            out,
            inputs_embeds.stride(0),
            inputs_embeds.stride(1),
            previous_hidden.stride(0),
            previous_hidden.stride(1),
            enorm_weight.stride(0),
            hnorm_weight.stride(0),
            out.stride(0),
            hidden,
            eps,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=8,
            num_stages=1,
        )
    return out


__all__ = ["fused_eh_norm"]

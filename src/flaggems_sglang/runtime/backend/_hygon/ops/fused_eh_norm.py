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

# Hygon vendor: the per-chip leaderboard puts 0.485 of the 0.71 mean gap on
# Haiguang alone (ours 10.92 vs the leader's 14.80) while the other chips
# are at parity, so only this platform needs a different shape. The generic
# computes both norm paths sequentially in one program per token; this
# vendor applies the page-table hygon playbook that gained +22% there - a
# direct 2D grid with no flattened task, one program per (token, path),
# halving the per-program serial work and doubling program-level
# parallelism. BLOCK, num_warps and num_stages stay at the generic values.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_eh_norm_hygon(
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
        _fused_eh_norm_hygon[(tokens, 2)](
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

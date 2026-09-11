# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

# Enflame vendor: byte-frozen clone of the generic that passed GCU at
# 2.53x (submission 12895). The round-7 generic reworks the op into a
# destination-driven gather behind a tiny inverse-routing scatter; the
# scatter stores through a loaded index, the exact lowering proven
# broken on this GCU stack (compute_src2dst rounds 1-5), so Enflame
# keeps this proven form.

import torch
import triton
import triton.language as tl


@triton.jit
def _deepep_permute(
    x,
    out,
    routes,
    tasks,
    tiles,
    topk,
    hidden,
    xs0,
    xs1,
    os0,
    os1,
    rs0,
    rs1,
    BLOCK: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        token_offset = (task // tiles).to(tl.int64)
        tile = task % tiles
        h = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        value = tl.load(x + token_offset * xs0 + h * xs1, h < hidden, other=0)
        value = value.to(out.dtype.element_ty)
        for slot in range(topk):
            dst = tl.load(
                routes + token_offset * rs0 + slot.to(tl.int64) * rs1
            ).to(tl.int64)
            if dst >= 0:
                tl.store(out + dst * os0 + h * os1, value, h < hidden)


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    assert input.ndim == gateup_input.ndim == src2dst.ndim == 2
    tokens, hidden = input.shape
    assert hidden_size == hidden and topk >= 0
    assert src2dst.shape == (tokens, topk)
    assert gateup_input.shape == (tokens * topk, hidden)
    assert src2dst.dtype in (torch.int32, torch.int64)
    out = gateup_input.clone()
    if tokens and topk and hidden:
        tiles = triton.cdiv(hidden, 512)
        tasks = tokens * tiles
        _deepep_permute[(min(tasks, 65535),)](
            input,
            out,
            src2dst,
            tasks,
            tiles,
            topk,
            hidden,
            *input.stride(),
            *out.stride(),
            *src2dst.stride(),
            BLOCK=512,
        )
    return out


__all__ = ["deepep_permute"]

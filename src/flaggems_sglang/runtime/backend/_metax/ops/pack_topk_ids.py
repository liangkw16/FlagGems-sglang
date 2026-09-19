# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for pack_topk_ids: BLOCK 2048 with the 8-warp pin -
# muxi reads 1.9 vs the leader's 2.4 (the e4 global warps-8 hurt only
# enflame; isolated here it targets the +26% muxi gap that alone
# covers the 0.8% distance to rank 1).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel"])
def _pack_topk_ids(topk_ids, topk_weights, out, numel, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        ids = tl.load(topk_ids + offs, m, other=0)
        w = tl.load(topk_weights + offs, m, other=0.0)
        # bf16 bits via the exact f32 widening: the bf16 pattern is the
        # high half of the widened float's bit pattern (the direct
        # f32->i16 bitcast is rejected by the XPU backend).
        wf = w.to(tl.bfloat16).to(tl.float32)
        bits = (
            (wf.to(tl.int32, bitcast=True) >> 16) & 0xFFFF
        )
        tl.store(out + offs, (ids << 16) | bits, m)


def pack_topk_ids(topk_ids, topk_weights):
    assert topk_ids.shape == topk_weights.shape
    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    out = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    if numel:
        _pack_topk_ids[(min(triton.cdiv(numel, 2048), 2048),)](
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK=2048,
            num_warps=8,
        )
    return out


__all__ = ["pack_topk_ids"]

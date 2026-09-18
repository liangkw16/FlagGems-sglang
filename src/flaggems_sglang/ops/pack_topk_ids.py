# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Pack (expert_id, weight) route pairs into one int32 - the FlashInfer
# TRT-LLM routed-MoE layout: out = (ids << 16) | (bf16 weight bits).
# The bf16 truncation is the contract (low halfword is the bf16 bit
# pattern, not a rounded fixed-point value). Pure elementwise int/bit
# work; all stores 1D.

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
        bits = (
            w.to(tl.bfloat16)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
            & 0xFFFF
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
        _pack_topk_ids[(min(triton.cdiv(numel, 1024), 2048),)](
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK=1024,
        )
    return out


__all__ = ["pack_topk_ids"]

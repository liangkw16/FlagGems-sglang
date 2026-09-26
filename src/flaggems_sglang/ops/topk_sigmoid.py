# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 108 topk_sigmoid: fused sigmoid MoE gate - sigmoid over router
# logits, descending top-k, optional renormalize, destination-passing
# into the preallocated fp32 weights and int32 ids. One program per
# token: BLOCK_E lanes score every expert, then k rounds of
# max+argmax+mask select the winners (ties excluded by the contract's
# fp32 logits); renormalize reads back the k written weights, rescales
# and rewrites them in the same kernel.

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_sigmoid_kernel(
    gating,
    topk_weights,
    topk_ids,
    num_experts,
    k,
    renorm,
    scale,
    gs0,
    ws0,
    is0,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    es = tl.arange(0, BLOCK_E)
    em = es < num_experts
    g = tl.load(gating + row * gs0 + es, mask=em, other=0.0).to(tl.float32)
    scores = tl.sigmoid(g)
    scores = tl.where(em, scores, float("-inf"))
    for i in range(0, k):
        v = tl.max(scores, axis=0)
        idx = tl.argmax(scores, axis=0)
        tl.store(topk_weights + row * ws0 + i, v)
        tl.store(topk_ids + row * is0 + i, idx)
        scores = tl.where(es == idx, float("-inf"), scores)
    if renorm:
        ks = tl.arange(0, 16)
        km = ks < k
        wv = tl.load(topk_weights + row * ws0 + ks, mask=km, other=0.0)
        total = tl.sum(wv, axis=0)
        tl.store(
            topk_weights + row * ws0 + ks,
            wv * scale / (total + 1e-20),
            mask=km,
        )


def topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor):
    num_tokens, num_experts = gating_output.shape
    k = topk_weights.shape[1]
    assert topk_ids.dtype == torch.int32
    if num_tokens:
        _topk_sigmoid_kernel[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            num_experts,
            k,
            renormalize,
            float(routed_scaling_factor),
            gating_output.stride(0),
            topk_weights.stride(0),
            topk_ids.stride(0),
            BLOCK_E=max(16, triton.next_power_of_2(num_experts)),
            num_warps=4,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]

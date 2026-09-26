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
    gs1,
    ws0,
    ws1,
    is0,
    is1,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    es = tl.arange(0, BLOCK_E)
    em = es < num_experts
    g = tl.load(gating + row * gs0 + es * gs1, mask=em, other=0.0).to(tl.float32)
    scores = tl.sigmoid(g)
    scores = tl.where(em, scores, float("-inf"))
    # rank each expert by how many others beat it: rank r means exactly
    # r experts score higher - a pure vector reduction with no per-round
    # scalar stores (the kunlun legalize pass rejected the argmax loop).
    # Ties break toward the lower expert id, matching torch.topk.
    higher = (scores[None, :] < scores[:, None]).to(tl.int32)
    eq_lower = (scores[None, :] == scores[:, None]) & (es[None, :] < es[:, None])
    rank = higher.sum(axis=1) + eq_lower.to(tl.int32).sum(axis=1)
    for i in range(0, k):
        pick = rank == i
        v = tl.max(tl.where(pick, scores, float("-inf")), axis=0)
        tl.store(topk_weights + row * ws0 + i * ws1, v)
        idx = tl.min(tl.where(pick, es, BLOCK_E), axis=0)
        tl.store(topk_ids + row * is0 + i * is1, idx)
    if renorm:
        ks = tl.arange(0, BLOCK_K)
        km = ks < k
        wv = tl.load(topk_weights + row * ws0 + ks * ws1, mask=km, other=0.0)
        total = tl.sum(wv, axis=0)
        tl.store(
            topk_weights + row * ws0 + ks * ws1,
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
            gating_output.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            BLOCK_E=max(16, triton.next_power_of_2(num_experts)),
            BLOCK_K=max(16, triton.next_power_of_2(k)),
            num_warps=4,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]

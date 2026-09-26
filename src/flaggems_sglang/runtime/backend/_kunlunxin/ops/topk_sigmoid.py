# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 108 topk_sigmoid - Kunlun XPU variant.
# The generic kernel materializes a (BLOCK_E, BLOCK_E) rank matrix and
# stores scalars inside the extraction loop; on XPU that exhausts
# uni_sram inside TritonXPULegalize (all cases failed to compile). This
# variant never builds a square matrix: rank is accumulated over column
# chunks of BLOCK_J experts (axis=1 reductions only), and the k winners
# are written with one masked 2-D scatter store - no per-round scalar
# stores anywhere.

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
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    es = tl.arange(0, BLOCK_E)
    em = es < num_experts
    g = tl.load(gating + row * gs0 + es * gs1, mask=em, other=0.0).to(tl.float32)
    scores = tl.sigmoid(g)
    scores = tl.where(em, scores, float("-inf"))
    rank = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for j0 in range(0, num_experts, BLOCK_J):
        js = j0 + tl.arange(0, BLOCK_J)
        jm = js < num_experts
        gj = tl.load(gating + row * gs0 + js * gs1, mask=jm, other=0.0).to(tl.float32)
        sj = tl.sigmoid(gj)
        sj = tl.where(jm, sj, float("-inf"))
        rs = tl.broadcast_to(scores[:, None], (BLOCK_E, BLOCK_J))
        cs = tl.broadcast_to(sj[None, :], (BLOCK_E, BLOCK_J))
        ri = tl.broadcast_to(es[:, None], (BLOCK_E, BLOCK_J))
        ci = tl.broadcast_to(js[None, :], (BLOCK_E, BLOCK_J))
        # count how many OTHER experts beat lane i (generic-kernel
        # direction): rank 0 is the winner. Padded other-lanes carry
        # -inf and never beat a real score; padded candidate lanes
        # accumulate rank >= num_experts and stay outside the k slots.
        beats = (cs > rs) | ((cs == rs) & (ci < ri))
        rank += tl.sum(beats.to(tl.int32), axis=1)
    ks = tl.arange(0, BLOCK_K)
    km = ks < k
    sel = (rank[:, None] == ks[None, :]) & km[None, :]
    w2 = tl.broadcast_to(scores[:, None], (BLOCK_E, BLOCK_K))
    i2 = tl.broadcast_to(es[:, None], (BLOCK_E, BLOCK_K))
    # the pointer must carry the same (BLOCK_E, BLOCK_K) shape as the
    # value; a (1, K) offset expression is not auto-broadcast by store
    zpad = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.int64)
    tl.store(topk_weights + row * ws0 + ks[None, :] * ws1 + zpad, w2, mask=sel)
    tl.store(topk_ids + row * is0 + ks[None, :] * is1 + zpad, i2, mask=sel)
    if renorm:
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
            BLOCK_J=16,
            BLOCK_K=max(16, triton.next_power_of_2(k)),
            num_warps=4,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]

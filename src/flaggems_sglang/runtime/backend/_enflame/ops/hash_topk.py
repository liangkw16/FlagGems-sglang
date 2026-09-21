# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for hash_topk: the tensor-index gather (indices from
# a tensor) fails GCU make_gcuir, so the wrapper stages it with torch
# gathers (the T49 precomputed-pos precedent) and the scoring core -
# sqrt(softplus), renormalisation, shared-expert append - runs as one
# Triton row kernel over the pre-gathered logits.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "num_routed"])
def _hash_topk_score(
    logits_g,
    out_weights,
    out_ids,
    eids,
    rows,
    num_routed,
    topk_real,
    nshared_real,
    inv_scale,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        width = topk_real + nshared_real
        offs = tl.arange(0, TOPK)
        mk = offs < topk_real
        logits = tl.load(logits_g + base * topk_real + offs, mk, other=0.0)
        sp = tl.where(
            logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))
        )
        w = tl.sqrt(sp)
        total = tl.sum(tl.where(mk, w, 0.0), axis=0)
        wn = w / total
        tl.store(out_weights + base * width + offs, wn, mk)
        ids = tl.load(eids + base * topk_real + offs, mk, other=0)
        tl.store(
            out_ids + base * width + offs, ids.to(tl.int32), mk
        )
        shared = tl.arange(0, NSHARED) + topk_real
        msh = (shared - topk_real) < nshared_real
        tl.store(
            out_weights + base * width + shared,
            tl.full((NSHARED,), 0.0, tl.float32) + inv_scale,
            msh,
        )
        tl.store(
            out_ids + base * width + shared,
            num_routed + (shared - topk_real),
            msh,
        )


def hash_topk(
    router_logits,
    input_ids,
    tid2eid,
    num_fused_shared_experts,
    routed_scaling_factor,
    scoring_func,
):
    assert scoring_func == "sqrtsoftplus"
    num_tokens, num_routed = router_logits.shape
    topk_routed = tid2eid.shape[1]
    expert_ids = tid2eid[input_ids.long()].long()
    logits_g = torch.gather(router_logits.float(), 1, expert_ids).contiguous()
    width = topk_routed + num_fused_shared_experts
    out_weights = torch.empty(
        (num_tokens, width), dtype=torch.float32,
        device=router_logits.device,
    )
    out_ids = torch.empty(
        (num_tokens, width), dtype=torch.int32,
        device=router_logits.device,
    )
    if num_tokens:
        _hash_topk_score[(min(num_tokens, 12),)](
            logits_g,
            out_weights,
            out_ids,
            expert_ids.to(torch.int32),
            num_tokens,
            num_routed,
            topk_routed,
            num_fused_shared_experts,
            1.0 / routed_scaling_factor,
            TOPK=triton.next_power_of_2(max(1, topk_routed)),
            NSHARED=triton.next_power_of_2(
                max(1, num_fused_shared_experts)
            ),
            num_warps=2,
            num_stages=3,
        )
    return out_weights, out_ids


__all__ = ["hash_topk"]

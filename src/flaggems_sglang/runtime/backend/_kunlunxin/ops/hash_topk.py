# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlun vendor for hash_topk: all three vector forms produced the same
# garbage-ids fingerprint on XPU (runtime-width addressing suspected),
# so this is the fully scalar serial form - one program, scalar loops
# over tokens and slots, no vector ops at all (the T84 precedent).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "num_routed", "topk", "nshared"])
def _hash_topk_scalar(
    router_logits,
    input_ids,
    tid2eid,
    out_weights,
    out_ids,
    rows,
    num_routed,
    topk,
    nshared,
    rs0,
    ts0,
    inv_scale,
):
    for m in range(tl.program_id(0), rows, tl.num_programs(0)):
        token = tl.load(input_ids + m).to(tl.int64)
        total = 0.0
        for j in range(0, topk):
            e = tl.load(tid2eid + token * ts0 + j).to(tl.int64)
            lg = tl.load(router_logits + m.to(tl.int64) * rs0 + e).to(
                tl.float32
            )
            sp = tl.where(lg > 20.0, lg, tl.log(1.0 + tl.exp(lg)))
            total += tl.sqrt(sp)
        for j in range(0, topk):
            e = tl.load(tid2eid + token * ts0 + j).to(tl.int64)
            lg = tl.load(router_logits + m.to(tl.int64) * rs0 + e).to(
                tl.float32
            )
            sp = tl.where(lg > 20.0, lg, tl.log(1.0 + tl.exp(lg)))
            w = tl.sqrt(sp) / total
            tl.store(out_weights + m * (topk + nshared) + j, w)
            tl.store(out_ids + m * (topk + nshared) + j, e.to(tl.int32))
        for j in range(0, nshared):
            tl.store(
                out_weights + m * (topk + nshared) + topk + j, inv_scale
            )
            tl.store(
                out_ids + m * (topk + nshared) + topk + j,
                num_routed + j,
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
    topk = tid2eid.shape[1]
    width = topk + num_fused_shared_experts
    out_weights = torch.empty(
        (num_tokens, width), dtype=torch.float32,
        device=router_logits.device,
    )
    out_ids = torch.empty(
        (num_tokens, width), dtype=torch.int32,
        device=router_logits.device,
    )
    if num_tokens:
        _hash_topk_scalar[(min(num_tokens, 512),)](
            router_logits,
            input_ids,
            tid2eid,
            out_weights,
            out_ids,
            num_tokens,
            num_routed,
            topk,
            num_fused_shared_experts,
            router_logits.stride(0),
            tid2eid.stride(0),
            1.0 / routed_scaling_factor,
        )
    return out_weights, out_ids


__all__ = ["hash_topk"]

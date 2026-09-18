# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Hash-routed MoE gate (DeepSeek-V4 style): the token's expert set is
# looked up in a precomputed tid2eid table (no top-k search), then
# route logits are gathered, scored with sqrt(softplus(.)), renormalised
# per row and the fused shared experts appended. Reference chains ~10
# eager kernels; one row-program kernel does it in a single pass. All
# stores 1D; softplus uses the F.softplus threshold-20 stable form.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "num_routed"])
def _hash_topk(
    router_logits,
    input_ids,
    tid2eid,
    out_weights,
    out_ids,
    rows,
    num_routed,
    topk_real,
    nshared_real,
    rs0,
    ts0,
    inv_scale,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        token = tl.load(input_ids + row).to(tl.int64)
        # masks use the REAL widths; TOPK/NSHARED are only pow2 pads
        # (masking with the pad reads past the table row and gathers
        # out-of-range expert ids).
        width = topk_real + nshared_real
        offs = tl.arange(0, TOPK)
        mk = offs < topk_real
        eids = tl.load(tid2eid + token * ts0 + offs, mk, other=0).to(
            tl.int64
        )
        logits = tl.load(
            router_logits + base * rs0 + eids, mk, other=0.0
        ).to(tl.float32)
        # F.softplus semantics: identity above the threshold, else
        # log1p(exp(x)) computed via the expm1-stable form.
        sp = tl.where(
            logits > 20.0,
            logits,
            tl.log(1.0 + tl.exp(logits)),
        )
        w = tl.sqrt(sp)
        total = tl.sum(tl.where(mk, w, 0.0), axis=0)
        wn = w / total
        tl.store(out_weights + base * width + offs, wn, mk)
        tl.store(out_ids + base * width + offs, eids, mk)
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
    assert tid2eid.ndim == 2 and input_ids.shape == (num_tokens,)
    assert tid2eid.dtype == torch.int32
    assert input_ids.dtype in (torch.int32, torch.int64)
    width = topk_routed + num_fused_shared_experts
    out_weights = torch.empty(
        (num_tokens, width), dtype=torch.float32, device=router_logits.device
    )
    out_ids = torch.empty(
        (num_tokens, width), dtype=torch.int32, device=router_logits.device
    )
    if num_tokens:
        _hash_topk[(min(num_tokens, 2048),)](
            router_logits,
            input_ids,
            tid2eid,
            out_weights,
            out_ids,
            num_tokens,
            num_routed,
            topk_routed,
            num_fused_shared_experts,
            router_logits.stride(0),
            tid2eid.stride(0),
            1.0 / routed_scaling_factor,
            TOPK=triton.next_power_of_2(max(1, topk_routed)),
            NSHARED=triton.next_power_of_2(
                max(1, num_fused_shared_experts)
            ),
        )
    return out_weights, out_ids


__all__ = ["hash_topk"]

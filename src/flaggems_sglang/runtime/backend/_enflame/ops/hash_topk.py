# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# GCU rejects vector-index gather. Read each routed slot through a
# scalar pointer instead of scanning the whole logits row per slot.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _hash_topk_match(
    router_logits,
    input_ids,
    tid2eid,
    out_weights,
    out_ids,
    rows,
    inv_scale,
    RS0: tl.constexpr,
    TS0: tl.constexpr,
    IDS_STEP: tl.constexpr,
    NROUTED: tl.constexpr,
    TOPK_REAL: tl.constexpr,
    NSHARED_REAL: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    NSHARED: tl.constexpr,
    IDS_I64: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        # token id as int32: i64 ids are reinterpreted as i32 pairs and
        # the little-endian low word is read (token ids are
        # non-negative and < vocab < 2**31, so the high word is zero).
        if IDS_I64:
            ids32 = input_ids.to(tl.pointer_type(tl.int32))
        else:
            ids32 = input_ids
        token = tl.load(ids32 + row * IDS_STEP)
        offs = tl.arange(0, TOPK)
        mk = offs < TOPK_REAL
        weights = tl.zeros((TOPK,), dtype=tl.float32)
        eids = tl.zeros((TOPK,), dtype=tl.int32)
        for slot in tl.static_range(TOPK_REAL):
            eid = tl.load(tid2eid + token * TS0 + slot)
            logit = tl.load(router_logits + row * RS0 + eid).to(tl.float32)
            sp = tl.where(logit > 20.0, logit, tl.log(1.0 + tl.exp(logit)))
            weights = tl.where(offs == slot, tl.sqrt(sp), weights)
            eids = tl.where(offs == slot, eid, eids)
        total = tl.sum(weights, axis=0)
        tl.store(out_weights + row * WIDTH + offs, weights / total, mk)
        tl.store(out_ids + row * WIDTH + offs, eids, mk)
        shared = tl.arange(0, NSHARED)
        msh = shared < NSHARED_REAL
        tl.store(
            out_weights + row * WIDTH + TOPK_REAL + shared,
            tl.full((NSHARED,), 0.0, tl.float32) + inv_scale,
            msh,
        )
        tl.store(
            out_ids + row * WIDTH + TOPK_REAL + shared,
            NROUTED + shared,
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
    # Layout normalisation only, never on the contiguous benchmark
    # path (T90 e6 round-2 contract: unit inner stride is required,
    # row-gapped strides run natively through the constexpr values).
    if router_logits.stride(1) != 1:
        router_logits = router_logits.contiguous()
    if tid2eid.stride(1) != 1:
        tid2eid = tid2eid.contiguous()
    ids_i64 = input_ids.dtype == torch.int64
    ids_step = input_ids.stride(0) * (2 if ids_i64 else 1)
    # int32 addressing domain (T90 e6 bound discipline). Bound proofs:
    # logits lanes <= (T-1)*RS0 + NROUTED-1 < T*RS0;
    # table lanes <= (vocab-1)*TS0 +
    # TOPK-1 < (vocab+1)*TS0 (pow2 pad TOPK <= 2*topk <= 2*TS0);
    # output lanes <= (T-1)*W + TOPK_REAL+NSHARED-1 < (T+2)*W
    # (pow2 pads <= 2x real widths); ids lanes < T*IDS_STEP + 1.
    assert num_tokens * router_logits.stride(0) < 2**31
    assert (tid2eid.shape[0] + 1) * tid2eid.stride(0) < 2**31
    assert (num_tokens + 2) * width < 2**31
    assert num_tokens * ids_step + 1 < 2**31
    out_weights = torch.empty(
        (num_tokens, width),
        dtype=torch.float32,
        device=router_logits.device,
    )
    out_ids = torch.empty(
        (num_tokens, width),
        dtype=torch.int32,
        device=router_logits.device,
    )
    if num_tokens:
        _hash_topk_match[(min(num_tokens, 12),)](
            router_logits,
            input_ids,
            tid2eid,
            out_weights,
            out_ids,
            num_tokens,
            1.0 / routed_scaling_factor,
            RS0=router_logits.stride(0),
            TS0=tid2eid.stride(0),
            IDS_STEP=ids_step,
            NROUTED=num_routed,
            TOPK_REAL=topk_routed,
            NSHARED_REAL=num_fused_shared_experts,
            WIDTH=width,
            TOPK=triton.next_power_of_2(max(1, topk_routed)),
            NSHARED=triton.next_power_of_2(max(1, num_fused_shared_experts)),
            IDS_I64=ids_i64,
            num_warps=2,
            num_stages=3,
        )
    return out_weights, out_ids


__all__ = ["hash_topk"]

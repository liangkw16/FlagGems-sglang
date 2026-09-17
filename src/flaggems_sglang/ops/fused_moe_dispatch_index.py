# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

import torch
import triton
import triton.language as tl

_E_TILE = 16


@triton.jit
def _fused_moe_dispatch_index(
    ids,
    src2dst,
    masked_m,
    m_max,
    num_toks,
    num_experts,
    BLOCK: tl.constexpr,
):
    # Warp-aggregated atomics (moe_align-style): one vector atomic per
    # (block, expert-tile) reserves the whole tile's tickets, and the
    # per-lane rank comes from an in-register one-hot cumsum - the
    # per-lane scalar atomic of the E13 pair form is the tianshu
    # bottleneck hypothesis (77 vs the field's 206-210). Signature and
    # grid-stride contract are unchanged (forced-grid poison test).
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    etiles = tl.arange(0, 16)
    for block in range(pid, tl.cdiv(num_toks, BLOCK), tl.num_programs(0)):
        base = block.to(tl.int64) * BLOCK
        offs = base + lanes
        mask = offs < num_toks
        expert = tl.load(ids + offs, mask=mask, other=-1)
        valid = mask & (expert >= 0)
        safe = tl.where(valid, expert, 0)
        acc_base = tl.zeros((BLOCK,), dtype=tl.int32)
        acc_rank = tl.zeros((BLOCK,), dtype=tl.int32)
        for e0 in range(0, num_experts, 16):
            e_ids = e0 + etiles
            e_ok = e_ids < num_experts
            onehot = (
                (expert[None, :] == e_ids[:, None]) & valid[None, :] & e_ok[:, None]
            )
            counts = tl.sum(onehot.to(tl.int32), axis=1)
            offsets = tl.atomic_add(masked_m + e_ids, counts, mask=e_ok, sem="relaxed")
            # Inclusive prefix along the block gives each lane its
            # within-tile rank (minus one for the exclusive ticket).
            pref = tl.cumsum(onehot.to(tl.int32), axis=1)
            sel = (e_ids[:, None] == safe[None, :]).to(tl.int32)
            lane_rank = tl.sum(pref * sel, axis=0)
            lane_base = tl.sum(offsets[:, None] * sel, axis=0)
            hit = valid & (expert >= e0) & (expert < e0 + 16)
            acc_rank = tl.where(hit, lane_rank - 1, acc_rank)
            acc_base = tl.where(hit, lane_base, acc_base)
        dst = safe.to(tl.int64) * m_max + (acc_base + acc_rank).to(tl.int64)
        tl.store(
            src2dst + offs,
            tl.where(valid, dst, 0).to(tl.int32),
            mask,
        )


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    assert topk_ids.ndim == 2
    assert topk_ids.dtype == torch.int32
    flat = topk_ids.reshape(-1)
    num_toks = flat.numel()
    assert num_local_experts >= 0
    masked_m = torch.zeros(num_local_experts, dtype=torch.int32, device=topk_ids.device)
    src2dst = torch.empty(num_toks, dtype=torch.int32, device=topk_ids.device)
    if num_toks:
        _fused_moe_dispatch_index[(min(triton.cdiv(num_toks, 256), 65535),)](
            flat,
            src2dst,
            masked_m,
            m_max,
            num_toks,
            num_local_experts,
            BLOCK=256,
        )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]

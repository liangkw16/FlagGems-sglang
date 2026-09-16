# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

import torch
import triton
import triton.language as tl


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
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK // 2)
    for block in range(pid, tl.cdiv(num_toks, BLOCK), tl.num_programs(0)):
        base = block.to(tl.int64) * BLOCK
        offs0 = base + lanes
        offs1 = offs0 + BLOCK // 2
        mask0 = offs0 < num_toks
        mask1 = offs1 < num_toks
        expert0 = tl.load(ids + offs0, mask0, other=-1)
        expert1 = tl.load(ids + offs1, mask1, other=-1)
        valid0 = mask0 & (expert0 >= 0)
        valid1 = mask1 & (expert1 >= 0)
        same = valid0 & valid1 & (expert0 == expert1)
        safe0 = tl.where(valid0, expert0, 0)
        safe1 = tl.where(valid1, expert1, 0)
        # Two striped routes share a logical lane; no cross-lane collective.
        ticket0 = tl.atomic_add(
            masked_m + safe0,
            1 + same.to(tl.int32),
            mask=valid0,
            sem="relaxed",
        )
        ticket0 = tl.where(valid0, ticket0, 0)
        issue1 = valid1 & ~same
        ticket1 = tl.atomic_add(
            masked_m + safe1, 1, mask=issue1, sem="relaxed"
        )
        ticket1 = tl.where(issue1, ticket1, 0)
        ticket1 = tl.where(same, ticket0 + 1, ticket1)
        dst0 = safe0 * m_max + ticket0
        dst1 = safe1 * m_max + ticket1
        tl.store(src2dst + offs0, tl.where(valid0, dst0, 0), mask0)
        tl.store(src2dst + offs1, tl.where(valid1, dst1, 0), mask1)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    assert topk_ids.ndim == 2
    assert topk_ids.dtype == torch.int32
    flat = topk_ids.reshape(-1)
    num_toks = flat.numel()
    masked_m = torch.zeros(
        num_local_experts, dtype=torch.int32, device=topk_ids.device
    )
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

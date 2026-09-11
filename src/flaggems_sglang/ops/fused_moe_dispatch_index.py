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
    for block in range(pid, tl.cdiv(num_toks, BLOCK), tl.num_programs(0)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_toks
        expert = tl.load(ids + offs.to(tl.int64), mask=mask, other=-1)
        valid = mask & (expert >= 0)
        expert_safe = tl.where(valid, expert, 0)
        offset = tl.atomic_add(masked_m + expert_safe, 1, mask=valid)
        dst = expert_safe * m_max + offset
        tl.store(src2dst + offs.to(tl.int64), dst, mask=valid)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    assert topk_ids.ndim == 2
    assert topk_ids.dtype == torch.int32
    flat = topk_ids.reshape(-1)
    num_toks = flat.numel()
    masked_m = torch.zeros(
        num_local_experts, dtype=torch.int32, device=topk_ids.device
    )
    src2dst = torch.zeros(num_toks, dtype=torch.int32, device=topk_ids.device)
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

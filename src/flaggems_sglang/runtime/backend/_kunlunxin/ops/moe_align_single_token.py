# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Scalar-serial vendor for moe_align_single_token (kunlun + haiguang):
# XPU fails to translate the vector rank forms and haiguang showed a
# 2/512 rank edge; this pure-scalar nested-loop form has no vector ops
# at all and computes ranks by direct pairwise comparison.

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_align_scalar(
    topk,
    sorted_ids,
    expert_ids,
    num_post,
    k_numel,
    total,
    block_sz,
    buf_numel,
):
    for base in range(0, buf_numel, 1024):
        o = base + tl.arange(0, 1024)
        v = tl.full((1024,), k_numel, dtype=tl.int32)
        tl.store(sorted_ids + o, v, o < buf_numel)
    for i in range(0, k_numel):
        rank = 0
        vi = tl.load(topk + i)
        for j in range(0, k_numel):
            vj = tl.load(topk + j)
            if vj < vi:
                rank += 1
        tl.store(expert_ids + rank, vi)
        tl.store(sorted_ids + rank * block_sz, i)
    tl.store(num_post, total)


def moe_align_single_token(topk_ids, block_size):
    assert topk_ids.ndim == 2 and topk_ids.shape[0] == 1
    topk = topk_ids.shape[1]
    device = topk_ids.device
    sorted_ids = torch.empty(
        topk * block_size, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(topk, dtype=torch.int32, device=device)
    num_post = torch.empty(1, dtype=torch.int32, device=device)
    _moe_align_scalar[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        topk,
        topk * block_size,
        block_size,
        sorted_ids.numel(),
    )
    return sorted_ids, expert_ids, num_post


__all__ = ["moe_align_single_token"]

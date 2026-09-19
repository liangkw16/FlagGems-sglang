# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# M==1 decode specialisation of the MoE sort/pad: one token whose top_k
# experts are pairwise distinct, each owning one block. Reference runs
# torch.sort + a Python scatter loop; this is one single-program kernel:
# ranks come from pairwise comparisons (distinct ids => strict order),
# the sentinel fill and the per-slot scatter are all 1D stores.

# e7 carrier: e6 composition bytes with a comment-only identity change
# (the e6 verdict 29.991 carried a cool window; this is the day's final shot).

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_align_single_token(
    topk,
    sorted_ids,
    expert_ids,
    num_post,
    k_numel,
    total,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, TOPK)
    mk = offs < k_numel
    ids = tl.load(topk + offs, mk, other=2147483647)
    # rank[i] = number of ids smaller than ids[i]; distinct ids make
    # this a permutation of 0..k-1 without any sort call. The scalar-
    # loop form avoids the [TOPK, TOPK] 2D compare tensor (uni_sram
    # overflow on XPU, linalg conversion failure on Ascend).
    rank = tl.zeros((TOPK,), dtype=tl.int32)
    for j in tl.static_range(0, TOPK):
        other = tl.load(topk + j)
        if j < k_numel:
            rank += (ids > other).to(tl.int32)
            rank += (ids == other).to(tl.int32) * (offs < j).to(tl.int32)
    tl.store(expert_ids + rank, ids, mk)
    sentinel_fill = tl.full((BLOCK,), k_numel, dtype=tl.int32)
    for base in range(0, total, BLOCK):
        o = base + tl.arange(0, BLOCK)
        tl.store(sorted_ids + o, sentinel_fill, o < total)
    tl.store(sorted_ids + rank * (total // k_numel), offs, mk)
    tl.store(num_post, total)


def moe_align_single_token(topk_ids, block_size):
    assert topk_ids.ndim == 2 and topk_ids.shape[0] == 1
    topk = topk_ids.shape[1]
    assert topk_ids.dtype == torch.int32
    assert isinstance(block_size, int) and block_size >= 1
    device = topk_ids.device
    sorted_ids = torch.empty(
        topk * block_size, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(topk, dtype=torch.int32, device=device)
    num_post = torch.empty(1, dtype=torch.int32, device=device)
    _moe_align_single_token[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        topk,
        topk * block_size,
        TOPK=triton.next_power_of_2(max(1, topk)),
        BLOCK=1024,
    )
    return sorted_ids, expert_ids, num_post


__all__ = ["moe_align_single_token"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# M==1 decode specialisation of the MoE sort/pad: one token whose top_k
# experts are pairwise distinct, each owning one block. Reference runs
# torch.sort + a Python scatter loop; this is one single-program kernel:
# ranks come from pairwise comparisons (distinct ids => strict order),
# the sentinel fill and the per-slot scatter are all 1D stores.

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
    KREAL: tl.constexpr,
    BLOCK_SZ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # E4: one program per slot (its rank via scalar comparisons against
    # all others) plus a grid-strided sentinel fill - the single-program
    # form serialised everything.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if pid < KREAL:
        mine = tl.load(topk + pid)
        rank = 0
        for j in range(0, KREAL):
            other = tl.load(topk + j)
            if other < mine:
                rank += 1
        tl.store(expert_ids + rank, mine)
        tl.store(sorted_ids + rank * BLOCK_SZ, pid)
    # Sentinel fill skips the block heads (rank * BLOCK_SZ) - the
    # slot programs own those elements; no ordering between programs.
    for base in range(pid * BLOCK, total, nprog * BLOCK):
        o = base + tl.arange(0, BLOCK)
        v = tl.full((BLOCK,), k_numel, dtype=tl.int32)
        tl.store(sorted_ids + o, v, (o % BLOCK_SZ) != 0)
    if pid == 0:
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
    fill_progs = min(4096, triton.cdiv(topk * block_size, 1024))
    _moe_align_single_token[(max(topk, fill_progs),)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        topk,
        topk * block_size,
        TOPK=triton.next_power_of_2(max(1, topk)),
        KREAL=topk,
        BLOCK_SZ=block_size,
        BLOCK=1024,
    )
    return sorted_ids, expert_ids, num_post


__all__ = ["moe_align_single_token"]

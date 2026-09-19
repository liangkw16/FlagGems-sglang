# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Hygon vendor for moe_align_single_token: single-program scalar-if
# form with a BLOCK 256 sentinel fill (the vector-rank generic was
# numerically correct but slow here; scalar serial ranks + wide fill
# split the difference; our hygon reads 22 vs the leader band 29-31).

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_align_hygon(
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
    if tl.program_id(0) == 0:
        for i in range(0, KREAL):
            mine = tl.load(topk + i)
            rank = 0
            for j in range(0, KREAL):
                other = tl.load(topk + j)
                if other < mine:
                    rank += 1
            tl.store(expert_ids + rank, mine)
            tl.store(sorted_ids + rank * BLOCK_SZ, i)
        tl.store(num_post, total)
    fill = tl.full((BLOCK,), k_numel, dtype=tl.int32)
    for base in range(tl.program_id(0) * BLOCK, total, tl.num_programs(0) * BLOCK):
        o = base + tl.arange(0, BLOCK)
        m = ((o % BLOCK_SZ) != 0) & (o < total)
        tl.store(sorted_ids + o, fill, m)


def moe_align_single_token(topk_ids, block_size):
    assert topk_ids.ndim == 2 and topk_ids.shape[0] == 1
    topk = topk_ids.shape[1]
    device = topk_ids.device
    sorted_ids = torch.empty(
        topk * block_size, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(topk, dtype=torch.int32, device=device)
    num_post = torch.empty(1, dtype=torch.int32, device=device)
    progs = max(1, min(1024, triton.cdiv(topk * block_size, 256)))
    _moe_align_hygon[(progs,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        topk,
        topk * block_size,
        TOPK=triton.next_power_of_2(max(1, topk)),
        KREAL=topk,
        BLOCK_SZ=block_size,
        BLOCK=256,
    )
    return sorted_ids, expert_ids, num_post


__all__ = ["moe_align_single_token"]

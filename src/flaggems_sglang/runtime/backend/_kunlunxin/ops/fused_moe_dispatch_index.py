# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Kunlunxin vendor, e8: the seven-crash seal was set while the failures
# were misread as infra windows. T65's unlock on 2026-09-15 isolated the
# actual compiler landmine in this family - masked vector loads reduced
# back to scalars via comparison masks + tl.sum (make_llir SIGABRT on
# this backend). Both _dispatch_counts ("hits") and _dispatch_ranks
# ("rank") used exactly that extraction form. E8 rewrites them as pure
# scalar walks: scalar loads, runtime branches and scalar load-modify-
# stores only - the forms T65 e6 passed with - while kernel2 (already
# scalar) and the three-kernel decomposition stay untouched. The
# isCloseOffsetAnalysis/isCloseUnrollControl kwargs are dropped so the
# default passes run again now that the aborting construct is gone.
# Branches are preferred over bool->int casts everywhere (no vector
# casts, no integer tl.where, no reductions of any kind).

import torch
import triton
import triton.language as tl

_BLOCK = 32
_E_TILE = 64


@triton.jit
def _dispatch_counts(
    ids,
    counts,
    n,
    num_experts,
    num_experts_pad,
    num_blocks,
    BLOCK: tl.constexpr,
    EXPERT_TILES: tl.constexpr,
):
    # E9: one program per expert with a register accumulator and a
    # single store per (block, expert) cell. E8's in-branch global
    # load-modify-store compiled fine but misexecuted on this backend
    # (masked_m off by small deltas - dropped increments); the
    # scalar-accumulator form matches kernel2 and the T65 e6-proven
    # branch-counting shape, with no memory RMW anywhere.
    for tile in range(EXPERT_TILES):
        e = tl.program_id(0) + tile * tl.num_programs(0)
        if EXPERT_TILES == 1 or e < num_experts:
            for block in range(0, num_blocks):
                hits = 0
                for j in range(0, BLOCK):
                    off = block * BLOCK + j
                    if off < n:
                        e_j = tl.load(ids + off.to(tl.int64)).to(tl.int32)
                        if e_j == e:
                            hits += 1
                tl.store(
                    counts + block.to(tl.int64) * num_experts_pad + e, hits
                )


@triton.jit
def _dispatch_prefix(
    counts,
    prefix,
    masked_m,
    num_experts,
    num_experts_pad,
    num_blocks,
    EXPERT_TILES: tl.constexpr,
):
    for tile in range(EXPERT_TILES):
        e = tl.program_id(0) + tile * tl.num_programs(0)
        if e < num_experts:
            run = 0
            for block in range(0, num_blocks):
                off = block.to(tl.int64) * num_experts_pad + e
                tl.store(prefix + off, run)
                run += tl.load(counts + off)
            tl.store(masked_m + e, run)


@triton.jit
def _dispatch_ranks(
    ids,
    prefix,
    src2dst,
    n,
    num_experts_pad,
    num_blocks,
    m_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        base = block.to(tl.int64) * num_experts_pad
        for j in range(0, BLOCK):
            off = block * BLOCK + j
            if off < n:
                e_j = tl.load(ids + off.to(tl.int64)).to(tl.int32)
                if e_j >= 0:
                    # In-bucket order is free (src2dst is compared as a
                    # per-bucket multiset), so the natural earlier-lane
                    # rank is one valid assignment. E8: scalar branch
                    # counting replaces the (e == e_j) & (idx < j)
                    # tl.sum reduction.
                    rank = 0
                    for k in range(0, j):
                        e_k = tl.load(
                            ids + (block * BLOCK + k).to(tl.int64)
                        ).to(tl.int32)
                        if e_k == e_j:
                            rank += 1
                    start = tl.load(prefix + base + e_j.to(tl.int64))
                    tl.store(
                        src2dst + off.to(tl.int64),
                        e_j * m_max + start + rank,
                    )


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    assert topk_ids.ndim == 2
    assert topk_ids.dtype == torch.int32
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    num_experts = num_local_experts
    masked_m = torch.zeros(
        num_experts, dtype=torch.int32, device=topk_ids.device
    )
    src2dst = torch.zeros(n, dtype=torch.int32, device=topk_ids.device)
    if n:
        num_experts_pad = triton.cdiv(num_experts, _E_TILE) * _E_TILE
        num_blocks = triton.cdiv(n, _BLOCK)
        expert_grid = max(1, min(num_experts, 65535))
        expert_tiles = triton.cdiv(num_experts, expert_grid)
        # zeros (not empty): the scalar counts walk only touches hit
        # slots, so untouched (block, expert) cells must read as zero.
        counts = torch.zeros(
            (num_blocks, num_experts_pad),
            dtype=torch.int32,
            device=flat.device,
        )
        prefix = torch.empty_like(counts)
        _dispatch_counts[(expert_grid,)](
            flat,
            counts,
            n,
            num_experts,
            num_experts_pad,
            num_blocks,
            BLOCK=_BLOCK,
            EXPERT_TILES=expert_tiles,
            num_warps=1,
            num_stages=1,
        )
        _dispatch_prefix[(expert_grid,)](
            counts,
            prefix,
            masked_m,
            num_experts,
            num_experts_pad,
            num_blocks,
            EXPERT_TILES=expert_tiles,
            num_warps=1,
            num_stages=1,
        )
        _dispatch_ranks[(min(num_blocks, 65535),)](
            flat,
            prefix,
            src2dst,
            n,
            num_experts_pad,
            num_blocks,
            m_max,
            BLOCK=_BLOCK,
            num_warps=1,
            num_stages=1,
        )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]

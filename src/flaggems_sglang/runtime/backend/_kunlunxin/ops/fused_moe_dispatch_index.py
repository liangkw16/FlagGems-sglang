# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Atomic-free fused_moe_dispatch_index for chips whose Triton stack cannot
# legalize masked tl.atomic_add (GCU300 PassManager failure) or whose runtime
# degrades under atomic cursors (Ascend). Three deterministic kernels replace
# the atomic cursor: per-block expert counts, an exclusive block prefix per
# expert, and an in-order per-lane rank with a contiguous store at the input
# position. E2 narrows every kernel to primitives this team has already
# passed on GCU300 after e1 still failed PassManager there and miscounted
# rank by one on Ascend: no atomics, no tl.cumsum (Kunlun/Enflame lowering
# poison), no [BLOCK, E] or [BLOCK, BLOCK] 2D broadcast reductions (e1's
# prime suspects), and no vector gather through a loaded index. kernel1
# counts each expert with a scalar expert loop over a 1D compare/sum;
# kernel3 walks the block lane by lane, extracts the lane's expert with a
# 1D sum-select, gathers its exclusive base with the scalar
# load-value-to-int64-addressing form deepep_permute passed at 2.53x, and
# stores one contiguous scalar - the same runtime-branch discipline as
# deepep_permute's `if dst >= 0`. Shared verbatim by the _ascend and
# _kunlunxin vendors.

import torch
import triton
import triton.language as tl

_BLOCK = 64
_E_TILE = 64


@triton.jit
def _dispatch_counts(
    ids,
    counts,
    n,
    num_experts,
    num_blocks,
    BLOCK: tl.constexpr,
    E_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        lane_mask = offs < n
        offs_rd = tl.where(lane_mask, offs, 0)
        e = tl.load(ids + offs_rd.to(tl.int64), lane_mask, other=-1).to(
            tl.int32
        )
        e64 = block.to(tl.int64) * num_experts
        for e0 in range(0, num_experts, E_TILE):
            for j in tl.static_range(E_TILE):
                expert = e0 + j
                if expert < num_experts:
                    hits = tl.sum((e == expert).to(tl.int32), axis=0)
                    tl.store(counts + e64 + expert, hits)


@triton.jit
def _dispatch_prefix(
    counts,
    prefix,
    masked_m,
    num_experts,
    num_blocks,
    E_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    for e0 in range(pid * E_TILE, num_experts, tl.num_programs(0) * E_TILE):
        lanes = e0 + tl.arange(0, E_TILE).to(tl.int64)
        mask = lanes < num_experts
        run = tl.zeros((E_TILE,), dtype=tl.int32)
        for block in range(0, num_blocks):
            c = tl.load(
                counts + block.to(tl.int64) * num_experts + lanes,
                mask=mask,
                other=0,
            )
            tl.store(
                prefix + block.to(tl.int64) * num_experts + lanes,
                run,
                mask=mask,
            )
            run += c
        tl.store(masked_m + lanes, run, mask=mask)


@triton.jit
def _dispatch_ranks(
    ids,
    prefix,
    src2dst,
    n,
    num_experts,
    num_blocks,
    m_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + idx
        lane_mask = offs < n
        offs_rd = tl.where(lane_mask, offs, 0)
        e = tl.load(ids + offs_rd.to(tl.int64), lane_mask, other=-1).to(
            tl.int32
        )
        block64 = block.to(tl.int64)
        for j in tl.static_range(BLOCK):
            off = block * BLOCK + j
            if off < n:
                e_j = tl.load(ids + off.to(tl.int64)).to(tl.int32)
                if e_j >= 0:
                    # In-bucket order is free (src2dst is compared as a
                    # per-bucket multiset), so the natural earlier-lane
                    # rank is one valid assignment.
                    rank = tl.sum(
                        ((e == e_j) & (idx < j)).to(tl.int32), axis=0
                    )
                    base = tl.load(
                        prefix + block64 * num_experts + e_j.to(tl.int64)
                    )
                    tl.store(
                        src2dst + off.to(tl.int64),
                        e_j * m_max + base + rank,
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
        num_blocks = triton.cdiv(n, _BLOCK)
        counts = torch.empty(
            (num_blocks, num_experts), dtype=torch.int32, device=flat.device
        )
        prefix = torch.empty_like(counts)
        _dispatch_counts[(min(num_blocks, 65535),)](
            flat,
            counts,
            n,
            num_experts,
            num_blocks,
            BLOCK=_BLOCK,
            E_TILE=_E_TILE,
        )
        _dispatch_prefix[(min(triton.cdiv(num_experts, _E_TILE), 65535),)](
            counts,
            prefix,
            masked_m,
            num_experts,
            num_blocks,
            E_TILE=_E_TILE,
        )
        _dispatch_ranks[(min(num_blocks, 65535),)](
            flat,
            prefix,
            src2dst,
            n,
            num_experts,
            num_blocks,
            m_max,
            BLOCK=_BLOCK,
        )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]

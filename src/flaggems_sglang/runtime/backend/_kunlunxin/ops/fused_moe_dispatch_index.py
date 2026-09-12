# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Atomic-free fused_moe_dispatch_index for chips whose Triton stack cannot
# legalize masked tl.atomic_add (GCU300 PassManager failure) or whose runtime
# degrades under atomic cursors (Ascend). Three deterministic kernels replace
# the atomic cursor: per-block expert counts, an exclusive block prefix per
# expert, and an in-block pairwise rank with a contiguous store at the input
# position. Only compare/sum/vector-load/gather/contiguous-store primitives
# are used - no atomics, no tl.cumsum (a known GCU compile poison), no
# scatter. Shared verbatim by the _ascend and _kunlunxin vendors.

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
        e = tl.load(ids + offs.to(tl.int64), offs < n, other=-1).to(tl.int32)
        e64 = block.to(tl.int64) * num_experts
        for e0 in range(0, num_experts, E_TILE):
            lanes = e0 + tl.arange(0, E_TILE).to(tl.int64)
            hits = tl.sum(e[:, None] == lanes[None, :].to(tl.int32), axis=0)
            tl.store(
                counts + e64 + lanes,
                hits.to(tl.int32),
                lanes < num_experts,
            )


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
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        lane_mask = offs < n
        e = tl.load(ids + offs.to(tl.int64), lane_mask, other=-1).to(tl.int32)
        # In-bucket rank of each valid lane within this block: count
        # same-expert lanes that appear earlier. Padding lanes (-1) match
        # each other but are never stored.
        same = e[:, None] == e[None, :]
        earlier = tl.arange(0, BLOCK)[None, :] < tl.arange(0, BLOCK)[:, None]
        local_rank = tl.sum((same & earlier).to(tl.int32), axis=1)
        base = tl.load(
            prefix + block.to(tl.int64) * num_experts + e.to(tl.int64),
            lane_mask & (e >= 0),
            other=0,
        )
        dst = e * m_max + base + local_rank
        tl.store(src2dst + offs.to(tl.int64), dst, lane_mask & (e >= 0))


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

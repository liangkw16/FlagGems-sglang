# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Atomic-free fused_moe_dispatch_index for chips whose Triton stack cannot
# legalize masked tl.atomic_add (GCU300 PassManager failure) or whose runtime
# degrades under atomic cursors (Ascend). Three deterministic kernels replace
# the atomic cursor: per-block expert counts, an exclusive block prefix per
# expert, and an in-order per-lane rank with a contiguous store at the input
# position. E3 removes the last loop-carried tensor: e1 and e2 both died in
# GCU300 PassManager and the only structure they always shared is kernel2's
# register-carried serial scan - the hand-rolled cumsum shape the batch-1/2
# retrospective flags as a Pipeline-failure family on Enflame/Kunlun. The
# prefix now runs one program per expert with a purely scalar accumulator
# (scalar loads/stores only, no masked scalar loads). kernel1 lost its tail
# branch by padding the counts/prefix rows to E_TILE multiples, and the int
# tl.where address clamps added in e2 are gone - integer tl.where has no
# passing precedent on GCU (clamp_position e1 evidence) and deepep_permute
# passes with plain masked vector loads. kernel3 keeps the e2 form: lane-by-
# lane walk, 1D rank reduction, scalar load-value-to-int64 gather (the
# deepep_permute 2.53x form), contiguous scalar stores under runtime
# branches. Shared verbatim by the _ascend and _kunlunxin vendors.

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
    num_experts_pad,
    num_blocks,
    BLOCK: tl.constexpr,
    E_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        e = tl.load(ids + offs.to(tl.int64), offs < n, other=-1).to(tl.int32)
        base = block.to(tl.int64) * num_experts_pad
        for expert in range(0, num_experts_pad):
            hits = tl.sum((e == expert).to(tl.int32), axis=0)
            tl.store(counts + base + expert, hits)


@triton.jit
def _dispatch_prefix(
    counts,
    prefix,
    masked_m,
    num_experts,
    num_experts_pad,
    num_blocks,
):
    e = tl.program_id(0)
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
    idx = tl.arange(0, BLOCK)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + idx
        e = tl.load(ids + offs.to(tl.int64), offs < n, other=-1).to(tl.int32)
        base = block.to(tl.int64) * num_experts_pad
        for j in range(0, BLOCK):
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
                    start = tl.load(prefix + base + e_j.to(tl.int64))
                    tl.store(
                        src2dst + off.to(tl.int64), e_j * m_max + start + rank
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
        counts = torch.empty(
            (num_blocks, num_experts_pad),
            dtype=torch.int32,
            device=flat.device,
        )
        prefix = torch.empty_like(counts)
        _dispatch_counts[(min(num_blocks, 65535),)](
            flat,
            counts,
            n,
            num_experts_pad,
            num_blocks,
            BLOCK=_BLOCK,
            E_TILE=_E_TILE,
        )
        _dispatch_prefix[(min(num_experts, 65535),)](
            counts,
            prefix,
            masked_m,
            num_experts,
            num_experts_pad,
            num_blocks,
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
        )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Atomic-free moe_align_block_size, shared verbatim by the _ascend,
# _enflame and _kunlunxin vendors. The generic's atomic cursor faults
# the Ascend device asynchronously (the error surfaces at the reference's
# synchronize - see the e5-e8 huawei 0.0 chain) and degrades GCU/XPU
# runtimes (the batch-5 fused_moe_dispatch_index lesson, whose no-atomic
# template this ports): per-32-block expert counts table, one program per
# expert for the exclusive block prefix, the tiny serial scan for aligned
# bases, the parallel sentinel/expert_ids fill, then an in-order per-lane
# rank placement (in-bucket order is free - the harness compares each
# expert range as a multiset).

import torch
import triton
import triton.language as tl

_BLOCK = 32
_E_TILE = 64


@triton.jit(do_not_specialize=["n", "num_routed", "num_experts_pad"])
def _mabs_counts(
    flat, counts, n, num_routed, num_experts_pad, num_blocks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        e = tl.load(flat + offs.to(tl.int64), offs < n, other=-1).to(
            tl.int32
        )
        base = block.to(tl.int64) * num_experts_pad
        for expert in range(0, num_routed):
            hits = tl.sum((e == expert).to(tl.int32), axis=0)
            tl.store(counts + base + expert, hits)


@triton.jit(
    do_not_specialize=[
        "num_routed", "num_experts_pad", "num_blocks",
    ]
)
def _mabs_prefix(
    counts, prefix, totals, num_routed, num_experts_pad, num_blocks,
):
    e = tl.program_id(0)
    if e < num_routed:
        run = 0
        for block in range(0, num_blocks):
            off = block.to(tl.int64) * num_experts_pad + e
            tl.store(prefix + off, run)
            run += tl.load(counts + off)
        tl.store(totals + e, run)


@triton.jit(do_not_specialize=["num_routed", "block_size"])
def _mabs_scan(
    totals, base, nblk, npost, num_routed, block_size,
):
    offset = 0
    for e in range(0, num_routed):
        n = tl.load(totals + e)
        tl.store(base + e, offset)
        aligned = ((n + block_size - 1) // block_size) * block_size
        tl.store(nblk + e, aligned // block_size)
        offset += aligned
    tl.store(npost, offset)


@triton.jit(do_not_specialize=["num_routed", "buf_numel"])
def _mabs_fill(
    expert_ids, sorted_ids, base, nblk, num_routed, buf_numel, sentinel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_routed:
        beg = tl.load(base + pid) // 1
        n = tl.load(nblk + pid)
        e = pid.to(tl.int32)
        b0 = beg
        for i in range(0, n, BLOCK):
            offs = i + tl.arange(0, BLOCK)
            tl.store(expert_ids + b0 + offs, e, offs < n)
    else:
        base_off = (pid - num_routed).to(tl.int64) * BLOCK
        offs = base_off + tl.arange(0, BLOCK)
        m = offs < buf_numel
        fill = tl.zeros((BLOCK,), dtype=tl.int32) + sentinel
        tl.store(sorted_ids + offs, fill, m)


@triton.jit(
    do_not_specialize=["n", "num_routed", "num_experts_pad", "num_blocks"]
)
def _mabs_place(
    flat, prefix, base, sorted_ids, n, num_routed, num_experts_pad,
    num_blocks, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    for block in range(pid, num_blocks, tl.num_programs(0)):
        offs = block * BLOCK + idx
        e = tl.load(flat + offs.to(tl.int64), offs < n, other=-1).to(
            tl.int32
        )
        for j in range(0, BLOCK):
            off = block * BLOCK + j
            if off < n:
                e_j = tl.load(flat + off.to(tl.int64)).to(tl.int32)
                if e_j >= 0:
                    if e_j < num_routed:
                        rank = tl.sum(
                            ((e == e_j) & (idx < j)).to(tl.int32), axis=0
                        )
                        pref = tl.load(
                            prefix + block.to(tl.int64) * num_experts_pad
                            + e_j.to(tl.int64)
                        )
                        b = tl.load(base + e_j.to(tl.int64))
                        tl.store(
                            sorted_ids + b.to(tl.int64) + pref + rank, off
                        )


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    assert topk_ids.ndim == 2
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    num_routed = num_experts - 1
    device = flat.device
    sorted_ids = torch.empty_like(sorted_token_ids)
    eids = expert_ids.clone()
    npost = torch.empty_like(num_tokens_post_pad)
    if n:
        num_experts_pad = triton.cdiv(num_routed, _E_TILE) * _E_TILE
        num_blocks = triton.cdiv(n, _BLOCK)
        counts = torch.empty(
            num_experts_pad * num_blocks, dtype=torch.int32, device=device
        )
        prefix = torch.empty_like(counts)
        totals = torch.empty(
            num_experts_pad, dtype=torch.int32, device=device
        )
        base = torch.empty_like(totals)
        nblk = torch.empty_like(totals)
        grid = min(num_blocks, 2048)
        _mabs_counts[(grid,)](
            flat, counts, n, num_routed, num_experts_pad, num_blocks,
            BLOCK=_BLOCK,
        )
        _mabs_prefix[(max(1, min(num_routed, 65535)),)](
            counts, prefix, totals, num_routed, num_experts_pad,
            num_blocks,
        )
        _mabs_scan[(1,)](
            totals, base, nblk, npost, num_routed, block_size,
        )
        _mabs_fill[
            (num_routed + triton.cdiv(sorted_ids.numel(), 1024),)
        ](
            eids,
            sorted_ids,
            base,
            nblk,
            num_routed,
            sorted_ids.numel(),
            n,
            BLOCK=1024,
        )
        _mabs_place[(grid,)](
            flat, prefix, base, sorted_ids, n, num_routed,
            num_experts_pad, num_blocks, BLOCK=_BLOCK,
        )
    else:
        sorted_ids.fill_(n)
        npost.fill_(0)
    return sorted_ids, eids, npost


__all__ = ["moe_align_block_size"]

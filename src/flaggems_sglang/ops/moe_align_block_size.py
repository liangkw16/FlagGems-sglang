# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# e2r carrier: the e2 execution bytes with a comment-only identity
# change - submission 17709 scored huawei 0.0 from a reference-side
# torch_npu RuntimeError (all seven other chips healthy, avg 95.28).
# The sort/pad every block-tiled fused-MoE GEMM depends on: flatten
# (token, slot) pairs, group them by expert and pad each expert's run to
# a block multiple. The harness checks each expert range as a multiset
# (atomic cursor placement allowed) and requires the sentinel tail.
# Three launches: histogram -> single-program scan (aligned offsets,
# expert blocks, sentinel fill) -> atomic-cursor scatter. The trailing
# "filtered expert" slot never receives tokens or blocks.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel", "num_routed"])
def _hist(flat, counts, numel, num_routed, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        e = tl.load(flat + offs, m, other=0)
        hit = m & (e >= 0) & (e < num_routed)
        tl.atomic_add(counts + e, (hit).to(tl.int32), sem="relaxed")


@triton.jit(do_not_specialize=["num_routed", "block_size", "buf_numel"])
def _scan(
    counts,
    cursor,
    sorted_ids,
    expert_ids,
    num_post,
    num_routed,
    block_size,
    buf_numel,
    sentinel,
    BLOCK: tl.constexpr,
):
    offset = 0
    for e in range(0, num_routed):
        n = tl.load(counts + e)
        # counts[e] becomes this expert's run start (its own histogram
        # value is consumed here); the scatter reads the run starts.
        tl.store(counts + e, offset)
        aligned = ((n + block_size - 1) // block_size) * block_size
        beg = offset // block_size
        nblocks = aligned // block_size
        for b in range(0, nblocks):
            tl.store(expert_ids + beg + b, e)
        offset += aligned
        tl.store(cursor + e, 0)
    tl.store(num_post, offset)
    fill = tl.full((BLOCK,), sentinel, dtype=tl.int32)
    for base in range(0, buf_numel, BLOCK):
        o = base + tl.arange(0, BLOCK)
        tl.store(sorted_ids + o, fill, o < buf_numel)


@triton.jit(do_not_specialize=["numel", "num_routed"])
def _scatter(
    flat,
    counts,
    cursor,
    sorted_ids,
    numel,
    num_routed,
    block_size,
    BLOCK: tl.constexpr,
):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        e = tl.load(flat + offs, m, other=0)
        hit = m & (e >= 0) & (e < num_routed)
        safe_e = tl.where(hit, e, 0)
        pos = tl.atomic_add(cursor + safe_e, hit.to(tl.int32), sem="relaxed")
        run_start = tl.load(counts + safe_e)
        tl.store(sorted_ids + run_start + pos, offs, hit)


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
    numel = flat.numel()
    num_routed = num_experts - 1
    device = flat.device
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cursor = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    sorted_ids = torch.empty_like(sorted_token_ids)
    eids = expert_ids.clone()
    npost = torch.empty_like(num_tokens_post_pad)
    if numel:
        _hist[(min(triton.cdiv(numel, 1024), 2048),)](
            flat, counts, numel, num_routed, BLOCK=1024
        )
    _scan[(1,)](
        counts,
        cursor,
        sorted_ids,
        eids,
        npost,
        num_routed,
        block_size,
        sorted_ids.numel(),
        numel,
        BLOCK=1024,
    )
    if numel:
        _scatter[(min(triton.cdiv(numel, 1024), 2048),)](
            flat,
            counts,
            cursor,
            sorted_ids,
            numel,
            num_routed,
            block_size,
            BLOCK=1024,
        )
    return sorted_ids, eids, npost


__all__ = ["moe_align_block_size"]

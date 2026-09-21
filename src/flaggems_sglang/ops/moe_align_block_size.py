# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# The sort/pad every block-tiled fused-MoE GEMM depends on: flatten
# (token, slot) pairs, group them by expert and pad each expert's run to
# a block multiple. The harness checks each expert range as a multiset
# (atomic cursor placement allowed) and requires the sentinel tail.
# The trailing "filtered expert" slot never receives tokens or blocks.
# e12 structure: the reference-style per-expert Python loop is long gone;
# this round collapses the launch count from five (memset + hist + scan +
# fill + scatter, plus an expert_ids clone) to three - one tiny state
# memset, one fused compute launch, one scatter - and writes the outputs
# in place into the caller's buffers (the reference's clone exists only
# to keep its own result pristine; the returned values are identical,
# including the undefined expert_ids tail). In the fused kernel the
# first P programs build the histogram with atomics and the last one
# through the ticket carries the whole scan as one vectorized
# tl.cumsum pass (the serial 256-iteration expert loop is gone), then
# fills expert_ids with per-lane variable-length runs; the remaining
# programs blanket sorted_ids with the sentinel in parallel. The scatter
# keeps its proven bytes: an atomic cursor claim per (token, slot).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel", "num_routed", "buf_numel", "hist_programs"])
def _compute(
    flat,
    sorted_ids,
    eids,
    npost_ptr,
    counts,
    cursor,
    ticket,
    numel,
    num_routed,
    block_size,
    buf_numel,
    hist_programs,
    sentinel,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < hist_programs:
        for base in range(
            pid * BLOCK_N, numel, hist_programs * BLOCK_N
        ):
            offs = base + tl.arange(0, BLOCK_N)
            m = offs < numel
            e = tl.load(flat + offs, m, other=0)
            hit = m & (e >= 0) & (e < num_routed)
            safe_e = tl.where(hit, e, 0)
            tl.atomic_add(counts + safe_e, (hit).to(tl.int32))
        # the last histogram program through the ticket runs the scan;
        # the acq_rel default on the ticket orders every other program's
        # counts atomics before this read
        done = tl.atomic_add(ticket, 1)
        if done == hist_programs - 1:
            lanes = tl.arange(0, BLOCK_E)
            lm = lanes < num_routed
            cnt = tl.load(counts + lanes, lm, other=0)
            aligned = (
                (cnt + block_size - 1) // block_size
            ) * block_size
            starts = tl.cumsum(aligned, 0) - aligned
            total = tl.sum(aligned, 0)
            # counts becomes the run starts (consumed by the scatter)
            tl.store(counts + lanes, starts, lm)
            # the caller's scratch buffer carries the scatter cursor
            tl.store(cursor + lanes, tl.zeros((BLOCK_E,), tl.int32), lm)
            nblk = aligned // block_size
            start_blk = starts // block_size
            max_nb = tl.max(nblk, 0)
            for i in range(0, max_nb):
                m = lm & (i < nblk)
                tl.store(eids + start_blk + i, lanes.to(tl.int32), m)
            tl.store(npost_ptr, total)
    else:
        fbase = (pid - hist_programs) * BLOCK_F
        foffs = fbase + tl.arange(0, BLOCK_F)
        fm = foffs < buf_numel
        fill = tl.zeros((BLOCK_F,), dtype=tl.int32) + sentinel
        tl.store(sorted_ids + foffs, fill, fm)


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
        pos = tl.atomic_add(cursor + safe_e, hit.to(tl.int32))
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
    buf_numel = sorted_token_ids.numel()
    # counts (zero-initialized for the atomic histogram) + the ticket in
    # one small memset; the scatter cursor lives in the caller's scratch
    state = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    counts = state[:num_experts]
    ticket = state[num_experts:]
    cursor = cumsum_buffer[:num_experts]
    hist_programs = max(1, min(triton.cdiv(numel, 1024), 256))
    fill_programs = triton.cdiv(buf_numel, 1024)
    _compute[(hist_programs + fill_programs,)](
        flat,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        counts,
        cursor,
        ticket,
        numel,
        num_routed,
        block_size,
        buf_numel,
        hist_programs,
        numel,
        BLOCK_N=1024,
        BLOCK_E=triton.next_power_of_2(max(1, num_routed)),
        BLOCK_F=1024,
    )
    if numel:
        _scatter[(min(triton.cdiv(numel, 1024), 2048),)](
            flat,
            counts,
            cursor,
            sorted_token_ids,
            numel,
            num_routed,
            block_size,
            BLOCK=1024,
        )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


__all__ = ["moe_align_block_size"]

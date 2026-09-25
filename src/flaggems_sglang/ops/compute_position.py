# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 6ed9843 kernels/ops/attention/position.py: the
# upstream kernel (and our s0/e2 port) recomputes the exclusive prefix
# serially inside every program - O(bs^2) aggregate work and a long
# serial dependency chain that capped tianshu at 585x while the field
# reads 4338x. e3 splits the op into two launches with no serial chain
# beyond a single-program vectorized scan: K1 computes the exclusive
# cumsum of extend_seq_lens into starts (chunked tl.cumsum with a
# running carry, int32), K2 is one program per request that reads its
# precomputed start and streams prefix_len + arange over its segment.
# e6 keeps two allocations per call (positions plus one shared int32
# [2*batch] buffer) while preserving wide starts exactly like the
# reference's wide-integer slicing: the first half of the shared
# buffer IS the int32 extend_start_loc contract output (same
# truncation the reference returns), and the second half carries the
# high 32 bits of each exclusive start. K2 recombines low|high into an
# int64 address so a cumulative start crossing 2^31 addresses the
# correct position instead of writing before the allocation (codex
# review P2); the prefix promotion keeps the a47602d2 boundary
# contract.
# re-roll carrier (comment-only): the e11r platform window read
# tianshu 2793-3209 while the e15 window hit 3298 with sha-identical
# generic bytes - water re-roll of the team-best set.
# e8 collapses the common batch <= 2048 range into ONE launch: every
# program derives its own exclusive start as a single masked int64
# vector sum over the whole lens row (vectorized O(batch) lanes, not
# the s0 serial chain), truncates it into the int32 contract output
# exactly like the reference's slicing, and streams its segment from
# the in-register int64 start - no second launch, no low/high split
# buffer. Larger batches keep the e6 two-launch bytes unchanged (the
# per-program whole-row load would go quadratic there).
# e17 (reserve candidate, review round 1): ONE allocation per call on
# both dispatch paths - the int64 positions buffer grows a small tail
# (ceil(batch/2) words on the fused path, batch words on the two-launch
# path) whose 1-D int32 view carries the contract start_loc (plus the
# wide hi half on the two-launch path). The regions are disjoint and
# the kernels are byte-identical, so only the per-call allocation count
# drops 2 -> 1: per-call allocs are priced heavily on
# tianshu/haiguang/huawei/kunlunxin (e6 3->2 allocs read +91% on
# kunlunxin, tianshu +8.6% in the same window; e11r gap vs c2flow:
# tianshu 2792.6 vs 4773.1, kunlunxin 106.1 vs 130.2, huawei 145.2 vs
# 612.0), while muxi (957.3 vs 961.0) and card_b (2105.47 vs 2105.78)
# sit at parity with the leader - alloc-insensitive chips that act as
# the rollback guards (any of them -5% fails the candidate).
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(
    lens, starts32, starts_hi, batch, BLOCK_BS: tl.constexpr
):
    # the running offsets stay int64; the int32 contract output
    # truncates exactly like the reference's returned int32 start
    # while the high half preserves the wide value for K2 addressing
    carry = tl.zeros((), dtype=tl.int64)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0).to(tl.int64)
        incl = tl.cumsum(seg, 0) + carry
        wide = incl - seg
        tl.store(starts32 + lanes, wide.to(tl.int32), m)
        tl.store(starts_hi + lanes, (wide >> 32).to(tl.int32), m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch"])
def _fill_positions(
    positions, starts32, starts_hi, prefix_lens, lens,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    lo = tl.load(starts32 + i).to(tl.int64) & 0xFFFFFFFF
    hi = tl.load(starts_hi + i).to(tl.int64)
    start = (hi << 32) | lo
    seq_len = tl.load(lens + i)
    prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
    for off in range(0, seq_len, BLOCK):
        o = off + tl.arange(0, BLOCK)
        tl.store(
            positions + start + o,
            prefix_len.to(tl.int64) + o,
            mask=o < seq_len,
        )


@triton.jit(do_not_specialize=["batch"])
def _fill_positions_fused(
    positions, start_loc, prefix_lens, lens, batch,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    i = tl.program_id(0)
    # own exclusive start: one masked int64 vector sum over the row -
    # wide enough for a cumulative start crossing 2^31 without the
    # e6 low/high split, and truncated into the int32 contract output
    # exactly like the reference's slicing
    lanes = tl.arange(0, BLOCK_BS)
    seg = tl.load(lens + lanes, lanes < batch, other=0).to(tl.int64)
    start = tl.sum(tl.where(lanes < i, seg, 0), 0)
    tl.store(start_loc + i, start.to(tl.int32))
    seq_len = tl.load(lens + i)
    prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
    for off in range(0, seq_len, BLOCK):
        o = off + tl.arange(0, BLOCK)
        tl.store(
            positions + start + o,
            prefix_len.to(tl.int64) + o,
            mask=o < seq_len,
        )


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    batch = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch
    assert extend_prefix_lens.ndim == extend_seq_lens.ndim == 1
    assert extend_prefix_lens.dtype == extend_seq_lens.dtype == torch.int32
    device = extend_seq_lens.device
    # one allocation per call: the int64 buffer is positions followed by
    # a tail whose 1-D int32 view carries the contract start_loc (and
    # the wide hi half on the two-launch path). The tail is padded to a
    # whole int64 word so an odd fused batch views cleanly; regions are
    # disjoint and the kernels see the same bytes as the two-alloc form.
    if batch <= 2048:
        tail_words = (batch + 1) // 2
    else:
        tail_words = batch
    buf = torch.empty(
        extend_seq_lens_sum + tail_words, dtype=torch.int64, device=device
    )
    positions = buf[:extend_seq_lens_sum]
    tail32 = buf[extend_seq_lens_sum:].view(torch.int32)
    if batch <= 2048:
        extend_start_loc = tail32[:batch]
        if batch:
            _fill_positions_fused[(batch,)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK=1024,
                BLOCK_BS=triton.next_power_of_2(max(batch, 16)),
            )
        return positions, extend_start_loc
    extend_start_loc = tail32[:batch]
    starts_hi = tail32[batch:]
    if batch:
        _starts_scan[(1,)](
            extend_seq_lens,
            extend_start_loc,
            starts_hi,
            batch,
            BLOCK_BS=triton.next_power_of_2(min(max(batch, 1), 8192)),
        )
        _fill_positions[(batch,)](
            positions,
            extend_start_loc,
            starts_hi,
            extend_prefix_lens,
            extend_seq_lens,
            HAS_PREFIX=has_prefix,
            BLOCK=1024,
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

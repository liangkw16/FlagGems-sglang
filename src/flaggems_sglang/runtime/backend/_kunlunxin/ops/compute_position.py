# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlunxin vendor for compute_position: the e6 two-launch bytes.
# The e8 generic fuses batch<=2048 into one launch whose per-program
# masked self-scan (tl.where(lanes < program_id, ...)) miscomputes on
# XPU - the scalar-broadcast-in-where family also seen on other ops -
# so this chip keeps the proven two-launch path: K1 single-program
# vectorized exclusive scan into low/high halves, K2 one program per
# request recombining the int64 start.
# (Below is the e6 generic source verbatim.)
# e3 splits the op into two launches with no serial chain
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
# e17 (reserve candidate, review round 1): ONE allocation per call -
# the positions int64 buffer grows a batch-word tail whose 1-D int32
# view is exactly the old int32[2*batch] wide-start buffer (low half =
# contract start_loc, high half = hi scratch). Kernel bytes unchanged;
# only the per-call alloc count drops 2 -> 1, the axis e6 already
# proved expensive on this chip (3 -> 2 allocs read +91% kunlunxin;
# e11r gap vs c2flow 106.1 vs 130.2). Stores stay 1-D and the view is
# 1-D, so the XPU 2D-broadcast packing bug does not apply; the 1-D
# int32 tail view on a non-zero storage offset still needs target-chip
# verification before release (target-runtime-unverified).
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


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    batch = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch
    assert extend_prefix_lens.ndim == extend_seq_lens.ndim == 1
    assert extend_prefix_lens.dtype == extend_seq_lens.dtype == torch.int32
    device = extend_seq_lens.device
    # one allocation per call: positions plus a batch-word tail whose
    # 1-D int32 view is exactly the old int32[2*batch] wide-start
    # buffer - low half = contract start_loc, high half = hi scratch.
    # Disjoint regions, identical kernel bytes.
    buf = torch.empty(
        extend_seq_lens_sum + batch, dtype=torch.int64, device=device
    )
    positions = buf[:extend_seq_lens_sum]
    tail32 = buf[extend_seq_lens_sum:].view(torch.int32)
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

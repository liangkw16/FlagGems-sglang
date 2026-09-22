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
# positions stays int64 with the arithmetic promoted before the store
# so prefixes near 2^31 stay exact (the a47602d2 boundary contract).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(
    lens, starts64, starts32, batch, BLOCK_BS: tl.constexpr
):
    # the running offsets stay int64 so addressing never wraps even
    # when a hypothetical batch exceeds the int32 domain; the int32
    # contract output truncates exactly like the reference's int32
    # exclusive cumsum (such batches are outside the reference's own
    # valid domain - its start tensor and slice bounds break too)
    carry = tl.zeros((), dtype=tl.int64)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0).to(tl.int64)
        incl = tl.cumsum(seg, 0) + carry
        tl.store(starts64 + lanes, incl - seg, m)
        tl.store(starts32 + lanes, (incl - seg).to(tl.int32), m)
        carry += tl.sum(seg, 0)


@triton.jit(
    do_not_specialize=["batch", "total"]
)
def _fill_positions(
    positions, starts64, prefix_lens, lens, batch, total,
    HAS_PREFIX: tl.constexpr, LOG_BS: tl.constexpr, BLOCK: tl.constexpr,
):
    # flat grid over positions: perfect load balance under skewed
    # request lengths (the per-request grid serialized long requests
    # into single programs) and a bounded program count on every chip
    # (Ascend degraded when the grid grew with bs). Each element finds
    # its owning request with a branchless binary search over the
    # int64 starts (stable once lo == hi, so fixed LOG_BS iterations
    # are safe even when the tree is shallower).
    for blk in range(
        tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)
    ):
        j = blk * BLOCK + tl.arange(0, BLOCK)
        jm = j < total
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.zeros((BLOCK,), dtype=tl.int32) + (batch - 1)
        for _ in tl.static_range(LOG_BS):
            mid = (lo + hi + 1) >> 1
            sv = tl.load(starts64 + mid, jm, other=0)
            take = jm & (sv <= j)
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        seg_start = tl.load(starts64 + lo, jm, other=0)
        prefix_len = tl.load(prefix_lens + lo, jm, other=0) if HAS_PREFIX else 0
        tl.store(
            positions + j, prefix_len.to(tl.int64) + (j - seg_start), jm
        )


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    batch = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch
    assert extend_prefix_lens.ndim == extend_seq_lens.ndim == 1
    assert extend_prefix_lens.dtype == extend_seq_lens.dtype == torch.int32
    device = extend_seq_lens.device
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=device
    )
    extend_start_loc = torch.empty(
        batch, dtype=torch.int32, device=device
    )
    starts64 = torch.empty(
        batch, dtype=torch.int64, device=device
    )
    if batch:
        _starts_scan[(1,)](
            extend_seq_lens,
            starts64,
            extend_start_loc,
            batch,
            BLOCK_BS=triton.next_power_of_2(min(max(batch, 1), 8192)),
        )
        _fill_positions[(min(triton.cdiv(extend_seq_lens_sum, 1024), 2048),)](
            positions,
            starts64,
            extend_prefix_lens,
            extend_seq_lens,
            batch,
            extend_seq_lens_sum,
            HAS_PREFIX=has_prefix,
            LOG_BS=max(1, (max(batch - 1, 1)).bit_length()),
            BLOCK=1024,
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

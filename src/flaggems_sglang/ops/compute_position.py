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
def _starts_scan(lens, starts, batch, BLOCK_BS: tl.constexpr):
    carry = tl.zeros((), dtype=tl.int32)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0)
        incl = tl.cumsum(seg, 0) + carry
        tl.store(starts + lanes, incl - seg, m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch"])
def _fill_positions(
    positions, starts, prefix_lens, lens,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    start = tl.load(starts + i)
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
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch, dtype=torch.int32, device=extend_seq_lens.device
    )
    if batch:
        _starts_scan[(1,)](
            extend_seq_lens,
            extend_start_loc,
            batch,
            BLOCK_BS=triton.next_power_of_2(min(max(batch, 1), 8192)),
        )
        _fill_positions[(batch,)](
            positions,
            extend_start_loc,
            extend_prefix_lens,
            extend_seq_lens,
            HAS_PREFIX=has_prefix,
            BLOCK=1024,
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

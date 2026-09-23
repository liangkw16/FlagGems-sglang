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
# e6 drops the separate int64 starts scratch buffer: the scan writes
# only the int32 extend_start_loc output, and K2 promotes that int32
# start (and the prefix) to int64 before addressing so prefixes near
# 2^31 stay exact (the a47602d2 boundary contract) - two allocations
# per call instead of three, every allocation being an output.
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(
    lens, starts32, batch, BLOCK_BS: tl.constexpr
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
        tl.store(starts32 + lanes, (incl - seg).to(tl.int32), m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch"])
def _fill_positions(
    positions, starts32, prefix_lens, lens,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    start = tl.load(starts32 + i).to(tl.int64)
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
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=device
    )
    extend_start_loc = torch.empty(
        batch, dtype=torch.int32, device=device
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

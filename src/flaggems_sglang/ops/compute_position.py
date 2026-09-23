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
# Small batches use the original single-launch request kernel; larger
# batches retain e6's two-launch scan/fill and shared int32[2*batch]
# buffer. Its high half preserves wide starts for K2 addressing past
# 2^31 (codex review P2); both paths promote prefix arithmetic to i64.
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _compute_position_single(
    positions,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    prefix_len = tl.load(prefix_lens + pid) if HAS_PREFIX else 0
    seq_len = tl.load(seq_lens + pid)
    start = tl.cast(0, tl.int64)
    for i in range(pid):
        start += tl.load(seq_lens + i)
    for tile in range(tl.cdiv(seq_len, BLOCK_TOKENS)):
        offset = tile * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        tl.store(
            positions + start + offset,
            prefix_len.to(tl.int64) + offset,
            mask=offset < seq_len,
        )
    tl.store(starts + pid, start)


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(lens, starts32, starts_hi, batch, BLOCK_BS: tl.constexpr):
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
    positions,
    starts32,
    starts_hi,
    prefix_lens,
    lens,
    HAS_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
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
    positions = torch.empty(extend_seq_lens_sum, dtype=torch.int64, device=device)
    if batch <= 512:
        extend_start_loc = torch.empty(batch, dtype=torch.int32, device=device)
        if batch:
            _compute_position_single[(batch,)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK_TOKENS=512,
            )
    else:
        wide_starts = torch.empty(2 * batch, dtype=torch.int32, device=device)
        extend_start_loc = wide_starts[:batch]
        starts_hi = wide_starts[batch:]
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

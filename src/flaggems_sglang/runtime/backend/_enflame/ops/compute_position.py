# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for compute_position, e3 two-launch fully-parallel
# form (see the generic's header): K1 is a single-program vectorized
# exclusive scan; K2 streams each request's segment from its
# precomputed start, grid-strided across at most 12 programs per the
# official gcu300 geometry (max_grid_size=(12,1,1), num_warps=2).
# The GCU type verifier rejects !tt.ptr<i64> at the signature level
# (T60 eight-round evidence), so the int64 output is written through
# an int32 view: a per-call host probe (arange(4) read back through
# the view - [0,1,2,3] means torch-gcu's narrowed packing, [0,0,1,0]
# the standard pairs; a module-level cache dict is rejected by the
# platform's code-safety scan, and ~100us is nothing against the
# ms-scale reference) dispatches between the packed-int32 kernel
# (element i at view index i, the only representable form under the
# narrowed storage) and the standard int64 kernel. Both compute paths
# are Triton; the NVIDIA proxy takes the int64 branch so the numeric
# matrix covers both sources.

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_NUM_WARPS = 2
_BLOCK = 2048


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(lens, starts, batch, BLOCK_BS: tl.constexpr):
    # int32 offsets by design: the GCU packed output cannot represent
    # values beyond the int31 domain anyway, and a batch whose total
    # exceeds 2^31 cannot allocate its positions tensor on this device
    carry = tl.zeros((), dtype=tl.int32)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0)
        incl = tl.cumsum(seg, 0) + carry
        tl.store(starts + lanes, incl - seg, m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch", "total"])
def _fill_positions_i64(
    positions, starts, prefix_lens, lens, batch, total,
    HAS_PREFIX: tl.constexpr, LOG_BS: tl.constexpr, BLOCK: tl.constexpr,
):
    # e4 flat form (see the generic): int64 stores, <=12 programs
    for blk in range(
        tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)
    ):
        j = blk * BLOCK + tl.arange(0, BLOCK)
        jm = j < total
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.zeros((BLOCK,), dtype=tl.int32) + (batch - 1)
        for _ in tl.static_range(LOG_BS):
            mid = (lo + hi + 1) >> 1
            sv = tl.load(starts + mid, jm, other=0)
            take = jm & (sv <= j)
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        seg_start = tl.load(starts + lo, jm, other=0)
        prefix_len = tl.load(prefix_lens + lo, jm, other=0) if HAS_PREFIX else 0
        tl.store(
            positions + j, prefix_len.to(tl.int64) + (j - seg_start), jm
        )


@triton.jit(do_not_specialize=["batch", "total"])
def _fill_positions_i32(
    positions_words, starts, prefix_lens, lens, batch, total,
    HAS_PREFIX: tl.constexpr, LOG_BS: tl.constexpr, BLOCK: tl.constexpr,
):
    # narrow-packed path: one int32 store per logical element at view
    # index j (the physical slot of element j under the packed layout)
    for blk in range(
        tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)
    ):
        j = blk * BLOCK + tl.arange(0, BLOCK)
        jm = j < total
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.zeros((BLOCK,), dtype=tl.int32) + (batch - 1)
        for _ in tl.static_range(LOG_BS):
            mid = (lo + hi + 1) >> 1
            sv = tl.load(starts + mid, jm, other=0)
            take = jm & (sv <= j)
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        seg_start = tl.load(starts + lo, jm, other=0)
        prefix_len = tl.load(prefix_lens + lo, jm, other=0) if HAS_PREFIX else 0
        tl.store(
            positions_words + j, prefix_len + (j - seg_start), jm
        )


def _int64_packed(device):
    # probed per call: a module-level cache dict is rejected by the
    # platform's code-safety scan (global mutable container), and one
    # tiny arange + readback is ~100us against a ms-scale reference
    probe = torch.arange(4, dtype=torch.int64, device=device)
    head = probe.view(torch.int32)[:4].tolist()
    return head == [0, 1, 2, 3]


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    assert extend_prefix_lens.ndim == extend_seq_lens.ndim == 1
    assert extend_prefix_lens.dtype == extend_seq_lens.dtype == torch.int32
    batch = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch
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
            num_warps=_NUM_WARPS,
        )
        log_bs = max(1, (max(batch - 1, 1)).bit_length())
        if _int64_packed(device):
            _fill_positions_i32[(_MAX_CTAS,)](
                positions.view(torch.int32),
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                extend_seq_lens_sum,
                HAS_PREFIX=has_prefix,
                LOG_BS=log_bs,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        else:
            _fill_positions_i64[(_MAX_CTAS,)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                extend_seq_lens_sum,
                HAS_PREFIX=has_prefix,
                LOG_BS=log_bs,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
    return positions, extend_start_loc


__all__ = ["compute_position"]

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
    carry = tl.zeros((), dtype=tl.int32)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0)
        incl = tl.cumsum(seg, 0) + carry
        tl.store(starts + lanes, incl - seg, m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch"])
def _fill_positions_i64(
    positions, starts, prefix_lens, lens, batch,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    for i in range(tl.program_id(0), batch, tl.num_programs(0)):
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


@triton.jit(do_not_specialize=["batch"])
def _fill_positions_i32(
    positions_words, starts, prefix_lens, lens, batch,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    # narrow-packed path: one int32 store per logical element at view
    # index i (the physical slot of element i under the packed layout)
    for i in range(tl.program_id(0), batch, tl.num_programs(0)):
        start = tl.load(starts + i)
        seq_len = tl.load(lens + i)
        prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
        for off in range(0, seq_len, BLOCK):
            o = off + tl.arange(0, BLOCK)
            tl.store(
                positions_words + start + o,
                prefix_len + o,
                mask=o < seq_len,
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
        if _int64_packed(device):
            _fill_positions_i32[(min(batch, _MAX_CTAS),)](
                positions.view(torch.int32),
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        else:
            _fill_positions_i64[(min(batch, _MAX_CTAS),)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
    return positions, extend_start_loc


__all__ = ["compute_position"]

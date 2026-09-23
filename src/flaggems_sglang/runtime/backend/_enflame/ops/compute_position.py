# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for compute_position, e3 two-launch fully-parallel
# form (see the generic's header): K1 is a single-program vectorized
# exclusive scan; K2 streams each request's segment from its
# precomputed start, grid-strided across at most 12 programs per the
# official gcu300 geometry (max_grid_size=(12,1,1), num_warps=2).
# The GCU type verifier rejects !tt.ptr<i64> at the signature level
# (T60 eight-round evidence), so the int64 output is written through
# an int32 view. The packed/standard layout decision happens ON
# DEVICE inside the fill kernel (arange(4) viewed as int32 words:
# [0,1,2,3] = torch-gcu's narrowed packing, [0,0,1,0] = standard
# pairs): the earlier per-call host readback (~100us D2H sync)
# dominated the op time on GCU, and narrowed tensors keep the int64
# logical metadata so numel/storage checks cannot distinguish the
# layouts. Both store forms stay Triton and no i64 pointer enters any
# signature, so the GCU300 verifier stays satisfied; the NVIDIA proxy
# takes the standard branch so the numeric matrix covers both forms.

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






@triton.jit(do_not_specialize=["batch"])
def _fill_dual_layout(
    positions_words,
    probe_words,
    starts,
    prefix_lens,
    lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # the packed/standard decision reads the probe CONTENT on device:
    # arange(4) viewed as int32 words is [0,1,2,3] under torch-gcu's
    # narrowed packing and [0,0,1,0] under the standard pairs layout -
    # the previous per-call host readback cost a ~100us D2H sync that
    # dominated the op on GCU. narrowed keeps the int64 logical
    # metadata, so numel/storage-size checks cannot distinguish the
    # layouts (codex review on the first e11 cut). both store forms
    # write through the int32 view only - no i64 pointer ever enters
    # the signature, so the GCU300 verifier stays satisfied.
    w = tl.load(probe_words + tl.arange(0, 4))
    packed = tl.sum((w == tl.arange(0, 4)).to(tl.int32), 0) == 4
    for i in range(tl.program_id(0), batch, tl.num_programs(0)):
        start = tl.load(starts + i)
        seq_len = tl.load(lens + i)
        prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
        for off in range(0, seq_len, BLOCK):
            o = off + tl.arange(0, BLOCK)
            m = o < seq_len
            idx = start + o
            if packed:
                tl.store(positions_words + idx, prefix_len + o, mask=m)
            else:
                # int32-only little-endian split: the wrapped sum IS the
                # low word; the carry-out of the unsigned addition is the
                # sign bit of the classic adder expression (a&b)|((a|b)&~s)
                # - NOT the sign of the sum (2^31 itself carries nothing).
                # int64 vector ALU bitops fail make_gcuir on gcu300
                lo = prefix_len + o
                carry = (prefix_len & o) | ((prefix_len | o) & ~lo)
                hi = (carry < 0).to(tl.int32)
                # widen only the address word index (arange/scalar-extsi
                # addressing has platform precedent on gcu300; the int64
                # vector ALU bitops were the compile wall)
                widx = 2 * idx.to(tl.int64)
                tl.store(positions_words + widx, lo, mask=m)
                tl.store(positions_words + widx + 1, hi, mask=m)


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
        probe = torch.arange(4, dtype=torch.int64, device=device)
        _fill_dual_layout[(min(max(batch, 1), _MAX_CTAS),)](
            positions.view(torch.int32),
            probe.view(torch.int32),
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

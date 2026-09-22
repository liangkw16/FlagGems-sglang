# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for compute_position: the e3 two-launch fully-parallel
# core (single-program vectorized exclusive scan + per-request fill)
# with the fill grid capped at 64 programs striding over requests.
# The uncapped per-request grid (bs programs, one small store span
# each) read huawei 152 against the field's 682 - the e2-era striped
# form held 217 at <=64 programs, so the CANN-side per-program cost is
# the bottleneck hypothesis this cap tests.

import torch
import triton
import triton.language as tl

_MAX_PROGRAMS = 64


@triton.jit(do_not_specialize=["batch"])
def _starts_scan(
    lens, starts64, starts32, batch, BLOCK_BS: tl.constexpr
):
    # the running offsets stay int64 so addressing never wraps even
    # when a hypothetical batch exceeds the int32 domain; the int32
    # contract output truncates exactly like the reference's int32
    # exclusive cumsum
    carry = tl.zeros((), dtype=tl.int64)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        m = lanes < batch
        seg = tl.load(lens + lanes, m, other=0).to(tl.int64)
        incl = tl.cumsum(seg, 0) + carry
        tl.store(starts64 + lanes, incl - seg, m)
        tl.store(starts32 + lanes, (incl - seg).to(tl.int32), m)
        carry += tl.sum(seg, 0)


@triton.jit(do_not_specialize=["batch"])
def _fill_positions(
    positions, starts64, prefix_lens, lens, batch,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
):
    for i in range(tl.program_id(0), batch, tl.num_programs(0)):
        start = tl.load(starts64 + i)
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
    starts64 = torch.empty(batch, dtype=torch.int64, device=device)
    if batch:
        _starts_scan[(1,)](
            extend_seq_lens,
            starts64,
            extend_start_loc,
            batch,
            BLOCK_BS=triton.next_power_of_2(min(max(batch, 1), 8192)),
        )
        _fill_positions[(min(batch, _MAX_PROGRAMS),)](
            positions,
            starts64,
            extend_prefix_lens,
            extend_seq_lens,
            batch,
            HAS_PREFIX=has_prefix,
            BLOCK=1024,
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

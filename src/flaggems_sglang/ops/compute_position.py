# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 6ed9843 kernels/ops/attention/position.py and the
# open PR #31284 striped kernel (head 454f6bb): extend-batch fused
# position + start-offset computation. The seed prefix-sum is bounded
# by min(bs, 64)^2 instead of the one-program-per-request O(bs^2);
# small batches keep the simple original shape. positions is int64 and
# start int32 - exact integer semantics, prefix may be empty.

import torch
import triton
import triton.language as tl

_STRIPE_MIN_BS = 64


@triton.jit(do_not_specialize=["batch"])
def _compute_position(
    positions,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # One program per request; the serial exclusive cumsum stays O(bs^2)
    # but only for the small-batch shape where it is launch-bound anyway
    # (the striped kernel below takes over at _STRIPE_MIN_BS).
    pid = tl.program_id(0).to(tl.int64)
    prefix_len = tl.load(prefix_lens + pid) if HAS_PREFIX else 0
    seq_len = tl.load(seq_lens + pid)
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(seq_lens + i)
    num_loop = tl.cdiv(seq_len, BLOCK_TOKENS)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_TOKENS) + i * BLOCK_TOKENS
        tl.store(
            positions + cumsum_start + offset,
            prefix_len.to(tl.int64) + offset,
            mask=offset < seq_len,
        )
    tl.store(starts + pid, cumsum_start)


@triton.jit(do_not_specialize=["batch", "rows_per_stripe"])
def _compute_position_striped(
    positions,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    ROWS_PER_STRIPE: tl.constexpr,
):
    # Large-batch path: <= 64 stripes, each fills ROWS_PER_STRIPE
    # consecutive rows after seeding its running offset once. The
    # serial prefix chain is bounded by (stripes-1) * rows_per_stripe,
    # so aggregate prefix work is O(min(bs, 64)^2).
    stripe = tl.program_id(0)
    row_begin = stripe * ROWS_PER_STRIPE
    row_end = tl.minimum(row_begin + ROWS_PER_STRIPE, batch)
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(row_begin):
        cumsum_start += tl.load(seq_lens + i)
    offsets = tl.arange(0, BLOCK_TOKENS)
    for row in range(row_begin, row_end):
        prefix_len = tl.load(prefix_lens + row) if HAS_PREFIX else 0
        seq_len = tl.load(seq_lens + row)
        tl.store(starts + row, cumsum_start)
        num_loop = tl.cdiv(seq_len, BLOCK_TOKENS)
        for tile in range(num_loop):
            token_offsets = offsets + tile * BLOCK_TOKENS
            tl.store(
                positions + cumsum_start + token_offsets,
                prefix_len.to(tl.int64) + token_offsets,
                mask=token_offsets < seq_len,
            )
        cumsum_start += seq_len


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
        if batch < _STRIPE_MIN_BS:
            _compute_position[(batch,)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK_TOKENS=512,
            )
        else:
            rows_per_stripe = 16
            num_stripes = triton.cdiv(batch, rows_per_stripe)
            if num_stripes > 64:
                num_stripes = 64
                rows_per_stripe = triton.cdiv(batch, num_stripes)
            _compute_position_striped[(num_stripes,)](
                positions,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK_TOKENS=512,
                ROWS_PER_STRIPE=rows_per_stripe,
            )
    return positions, extend_start_loc


__all__ = ["compute_position"]

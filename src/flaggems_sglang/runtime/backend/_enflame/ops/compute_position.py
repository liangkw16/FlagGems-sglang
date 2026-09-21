# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for compute_position. The GCU type verifier rejects
# !tt.ptr<i64> at the signature level (T60 eight-round evidence), so the
# kernel signature must stay i64-free. torch-gcu allocates int64 tensors
# physically narrowed - 4N bytes for N logical elements while the storage
# metadata still claims 8N (t60-narrow-storage audit at torch-gcu f17a922,
# gcu_empty_tensor.cpp:50-64) - and T60 E8's two-word stores mismatched
# 100% on the platform, which fits the narrow packing: the physical int32
# slot of logical element i sits at int32-view index i. The wrapper probes
# the live layout once per process with torch (arange(4) read back through
# the int32 view: [0, 1, 2, 3] means packed, [0, 0, 1, 0] means standard
# pairs) and dispatches between two Triton kernels; both compute paths are
# Triton, no PyTorch fallback of the operator itself. On the narrow layout
# the packed int32 write is the only representable form - values beyond
# int32 cannot exist in the platform's own reference on this chip either,
# because the same narrowed storage backs torch.arange(int64). GCU launch
# geometry follows the vendor guidance: grid capped at 24 programs and a
# wide BLOCK_TOKENS for the streaming fill.

import torch
import triton
import triton.language as tl

_STRIPE_MIN_BS = 64
# FlagGems _enflame gcu300 codegen: max_grid_size=(12, 1, 1) with the
# launch clamped via grid-stride loops, and enflame_heuristics_for_num_warps
# pins num_warps=2 (the GCU warp is a vector engine, not SIMT 32 lanes).
_MAX_PROGRAMS = 12
_NUM_WARPS = 2
_BLOCK_TOKENS = 2048


@triton.jit(do_not_specialize=["batch"])
def _compute_position_i64(
    positions,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # Standard-layout path (also the form the NVIDIA proxy exercises):
    # int64 stores, arithmetic promoted before the store so prefixes near
    # 2**31 stay exact.
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
def _compute_position_striped_i64(
    positions,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    rows_per_stripe,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    ROWS_PER_STRIPE: tl.constexpr,
):
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


@triton.jit(do_not_specialize=["batch"])
def _compute_position_i32(
    positions_words,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # Narrow-packed path: one int32 store per logical element at view
    # index i (the physical slot of element i under the packed layout).
    pid = tl.program_id(0)
    prefix_len = tl.load(prefix_lens + pid) if HAS_PREFIX else 0
    seq_len = tl.load(seq_lens + pid)
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(seq_lens + i)
    num_loop = tl.cdiv(seq_len, BLOCK_TOKENS)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_TOKENS) + i * BLOCK_TOKENS
        tl.store(
            positions_words + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(starts + pid, cumsum_start)


@triton.jit(do_not_specialize=["batch", "rows_per_stripe"])
def _compute_position_striped_i32(
    positions_words,
    starts,
    prefix_lens,
    seq_lens,
    batch,
    rows_per_stripe,
    HAS_PREFIX: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    ROWS_PER_STRIPE: tl.constexpr,
):
    stripe = tl.program_id(0)
    row_begin = stripe * ROWS_PER_STRIPE
    row_end = tl.minimum(row_begin + ROWS_PER_STRIPE, batch)
    cumsum_start = 0
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
                positions_words + cumsum_start + token_offsets,
                prefix_len + token_offsets,
                mask=token_offsets < seq_len,
            )
        cumsum_start += seq_len


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
        packed = _int64_packed(device)
        if packed:
            positions_words = positions.view(torch.int32)
            small, striped = _compute_position_i32, _compute_position_striped_i32
        else:
            positions_words = positions
            small, striped = _compute_position_i64, _compute_position_striped_i64
        if batch < _STRIPE_MIN_BS:
            small[(batch,)](
                positions_words,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                HAS_PREFIX=has_prefix,
                BLOCK_TOKENS=_BLOCK_TOKENS,
                num_warps=_NUM_WARPS,
            )
        else:
            rows_per_stripe = 16
            num_stripes = triton.cdiv(batch, rows_per_stripe)
            if num_stripes > _MAX_PROGRAMS:
                num_stripes = _MAX_PROGRAMS
                rows_per_stripe = triton.cdiv(batch, num_stripes)
            striped[(num_stripes,)](
                positions_words,
                extend_start_loc,
                extend_prefix_lens,
                extend_seq_lens,
                batch,
                rows_per_stripe,
                HAS_PREFIX=has_prefix,
                BLOCK_TOKENS=_BLOCK_TOKENS,
                ROWS_PER_STRIPE=rows_per_stripe,
                num_warps=_NUM_WARPS,
            )
    return positions, extend_start_loc


__all__ = ["compute_position"]

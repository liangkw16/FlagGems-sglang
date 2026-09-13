# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/jit/csrc/elementwise/concat_mla.cuh.

# Kunlunxin vendor, round 5. Four deterministic-garbage rounds (s0 flat
# 2D tile, s0r/s0r2 carriers, BH=16, and the two-clamped-loads row form)
# all share one addressing recipe the FlagTree #1147 analysis condemns:
# a single padded range over the non-power-of-2 row width (192 -> 256
# with a dead lane zone) feeding two masked stores off one base, plus
# the rope reload whose offset pattern repeats across every head of a
# token - the stride-0 broadcast read that OffsetAnalysis misjudges as
# Continuous before bursting past the wrap. This round adopts the
# official FlagGems concat_and_cache_mla structure exactly: two fully
# independent 1D segment loops with their own power-of-two blocks (the
# k segment at 128 lanes, the rope segment at 64), no shared base
# expression, no dead lanes, no in-kernel broadcast.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _concat_mla_k_segments(
    nope,
    rope,
    out,
    rows,
    heads,
    nd,
    rd,
    ns0,
    ns1,
    ns2,
    rs0,
    rs2,
    os1,
    os2,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        token = row64 // heads
        head = row64 % heads
        base = row64 * os1
        src = token * ns0 + head * ns1
        for i in range(0, nd, BLOCK_N):
            offs = (i + tl.arange(0, BLOCK_N)).to(tl.int64)
            m = offs < nd
            no = tl.load(nope + src + offs * ns2, m, other=0)
            tl.store(out + base + offs * os2, no, m)
        rbase = token * rs0
        for j in range(0, rd, BLOCK_R):
            offs = (j + tl.arange(0, BLOCK_R)).to(tl.int64)
            m = offs < rd
            ro = tl.load(rope + rbase + offs * rs2, m, other=0)
            tl.store(out + base + (nd + offs) * os2, ro, m)


def concat_mla_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        rows = tokens * heads
        _concat_mla_k_segments[(min(rows, _MAX_GRID),)](
            k_nope,
            k_rope,
            out,
            rows,
            heads,
            nd,
            rd,
            *k_nope.stride(),
            k_rope.stride(0),
            k_rope.stride(2),
            out.stride(1),
            out.stride(2),
            BLOCK_N=triton.next_power_of_2(max(1, nd)),
            BLOCK_R=triton.next_power_of_2(max(1, rd)),
        )
    return out


__all__ = ["concat_mla_k"]

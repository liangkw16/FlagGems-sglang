# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

# E4: BLOCK 1024 -> 2048 (e3 enflame 3.01, +19% - ladder continues).
# E3: BLOCK 512 -> 1024 - c2flow's no-anomaly shape reads ~2x ours on
# every chip (tianshu 34 vs 16, enflame 12.3 vs 2.5); wider hidden tiles
# are the cheapest broad axis before any structural change.
# Enflame vendor: byte-frozen clone of the generic that passed GCU at
# 2.53x (submission 12895). The round-7 generic reworks the op into a
# destination-driven gather behind a tiny inverse-routing scatter; the
# scatter stores through a loaded index, the exact lowering proven
# broken on this GCU stack (compute_src2dst rounds 1-5), so Enflame
# keeps this proven form.

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_SMALL_BYTES = 20 * 1024 * 1024


@triton.jit
def _deepep_permute(
    x,
    out,
    routes,
    tasks,
    tiles,
    topk,
    hidden,
    xs0,
    xs1,
    os0,
    os1,
    rs0,
    rs1,
    BLOCK: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        token_offset = (task // tiles).to(tl.int64)
        tile = task % tiles
        h = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        value = tl.load(x + token_offset * xs0 + h * xs1, h < hidden, other=0)
        value = value.to(out.dtype.element_ty)
        for slot in range(topk):
            dst = tl.load(routes + token_offset * rs0 + slot.to(tl.int64) * rs1).to(
                tl.int64
            )
            if dst >= 0:
                tl.store(out + dst * os0 + h * os1, value, h < hidden)


@triton.jit
def _build_inv(
    routes,
    inv,
    tokens,
    topk,
    rs0,
    rs1,
    TOPK: tl.constexpr,
):
    # E10: record the source token per destination row so the wide path
    # becomes a pure gather (single gather + single masked store is the
    # GCU-proven form, T63-e1); the scalar scatter store here matches
    # the legacy kernel's proven store-through-loaded-index lowering.
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        base = routes + token * rs0
        for slot in tl.static_range(TOPK):
            dst = tl.load(base + slot * rs1)
            if dst >= 0:
                tl.store(inv + dst, token)


@triton.jit
def _gather_rows(
    x,
    gateup,
    inv,
    out,
    tasks,
    tiles,
    hidden,
    xs0,
    xs1,
    gs0,
    gs1,
    os0,
    os1,
    BLOCK: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        row = (task // tiles).to(tl.int64)
        tile = task % tiles
        h = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        src = tl.load(inv + row).to(tl.int64)
        if src >= 0:
            value = tl.load(x + src * xs0 + h * xs1, h < hidden, other=0)
            tl.store(
                out + row * os0 + h * os1,
                value.to(out.dtype.element_ty),
                h < hidden,
            )
        else:
            value = tl.load(gateup + row * gs0 + h * gs1, h < hidden, other=0)
            tl.store(out + row * os0 + h * os1, value, h < hidden)


def deepep_permute(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    assert input.ndim == gateup_input.ndim == src2dst.ndim == 2
    tokens, hidden = input.shape
    assert hidden_size == hidden and topk >= 0
    assert src2dst.shape == (tokens, topk)
    assert gateup_input.shape == (tokens * topk, hidden)
    assert src2dst.dtype in (torch.int32, torch.int64)
    if not (tokens and topk and hidden):
        return gateup_input.clone()
    if gateup_input.numel() * gateup_input.element_size() <= _SMALL_BYTES:
        # Launch-overhead regime keeps the proven clone form (see the
        # generic module's _scatter_legacy note).
        out = gateup_input.clone()
        tiles = triton.cdiv(hidden, _BLOCK)
        tasks = tokens * tiles
        # C1 (R2 pre-registered): cap the grid at the GCU's physical
        # scheduling width instead of 65535 - this stack's launch overhead
        # cannot be hidden and oversized grids are pure scheduling cost
        # (skill hard-facts: 24 SIPs). The kernel body already strides.
        _deepep_permute[(min(tasks, 24),)](
            input,
            out,
            src2dst,
            tasks,
            tiles,
            topk,
            hidden,
            *input.stride(),
            *out.stride(),
            *src2dst.stride(),
            BLOCK=_BLOCK,
        )
        return out
    # E10 wide path: clone-free inverse-map gather under the GCU recipe
    # (persistent <=24 grid, num_stages=3 pingpong, no num_warps pin).
    out = torch.empty_like(gateup_input)
    rows = tokens * topk
    inv = torch.full((rows,), -1, dtype=torch.int32, device=gateup_input.device)
    _build_inv[(min(tokens, 24),)](
        src2dst,
        inv,
        tokens,
        topk,
        *src2dst.stride(),
        TOPK=topk,
    )
    tiles = triton.cdiv(hidden, 512)
    tasks = rows * tiles
    _gather_rows[(min(tasks, 24),)](
        input,
        gateup_input,
        inv,
        out,
        tasks,
        tiles,
        hidden,
        *input.stride(),
        *gateup_input.stride(),
        *out.stride(),
        BLOCK=512,
        num_stages=3,
    )
    return out


__all__ = ["deepep_permute"]

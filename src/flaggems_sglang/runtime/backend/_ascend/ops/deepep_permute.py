# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor: clone-free three-pass structure (generic E7) with the
# launch kept close to the physical Vector Core count. Ascend's vector
# operator guidance ("data/vendor-backends/ascend/vector_operator.md")
# states that GPU-style huge grids cause repeated dispatch overhead on
# NPUs; every kernel therefore uses num_vectorcore programs and strides
# over its tasks in an inner loop (decode_attention template).

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

_DEFAULT_VECTOR_CORES = 40

# Below this many gateup bytes the wrapper keeps the clone form; see
# the generic module's _scatter_legacy note.
_SMALL_BYTES = 20 * 1024 * 1024


def _get_num_vector_cores(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties.get("num_vectorcore", _DEFAULT_VECTOR_CORES))


@triton.jit
def _scatter_legacy(
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
    # Clone-form kernel for the launch-overhead regime; see generic.
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
    # E9 gather layout: record the source token for every destination
    # row; the main pass then only gathers rows and stores contiguously
    # (no scatter stores - Ascend vector cores favor gathered loads).
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
    workers = _get_num_vector_cores(gateup_input.device.index)
    if gateup_input.numel() * gateup_input.element_size() <= _SMALL_BYTES:
        out = gateup_input.clone()
        tiles = triton.cdiv(hidden, 512)
        tasks = tokens * tiles
        _scatter_legacy[(min(tasks, workers),)](
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
            BLOCK=512,
        )
        return out
    out = torch.empty_like(gateup_input)
    rows = tokens * topk
    inv = torch.full((rows,), -1, dtype=torch.int32, device=gateup_input.device)
    _build_inv[(min(tokens, workers),)](
        src2dst,
        inv,
        tokens,
        topk,
        *src2dst.stride(),
        TOPK=topk,
    )
    tiles = triton.cdiv(hidden, 2048)
    tasks = rows * tiles
    _gather_rows[(min(tasks, workers),)](
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
        BLOCK=2048,
    )
    return out


__all__ = ["deepep_permute"]

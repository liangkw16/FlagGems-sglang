# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Materialize both eager bf16 intermediates across kernel boundaries on XPU.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _round_scale(
    scale,
    indices,
    one_plus,
    rows,
    cs0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        idx = tl.load(indices + row).to(tl.int64)
        base = row.to(tl.int64)
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            sc = tl.load(scale + idx * cs0 + offs, m, other=0).to(tl.float32)
            rounded = (1.0 + sc).to(tl.bfloat16, fp_downcast_rounding="rtne")
            tl.store(one_plus + base * HDIM + offs, rounded, m)


@triton.jit(do_not_specialize=["rows"])
def _scale_rows(
    x,
    one_plus,
    out,
    rows,
    xs0,
    os0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            xv = tl.load(x + base * xs0 + offs, m, other=0).to(tl.float32)
            factor = tl.load(one_plus + base * HDIM + offs, m, other=0).to(
                tl.float32
            )
            scaled = (xv * factor).to(tl.bfloat16, fp_downcast_rounding="rtne")
            tl.store(out + base * os0 + offs, scaled, m)


@triton.jit(do_not_specialize=["rows"])
def _shift_rows(
    shift,
    indices,
    out,
    rows,
    ss0,
    os0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        idx = tl.load(indices + row).to(tl.int64)
        base = row.to(tl.int64)
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            scaled = tl.load(out + base * os0 + offs, m, other=0).to(
                tl.float32
            )
            sh = tl.load(shift + idx * ss0 + offs, m, other=0).to(tl.float32)
            rounded = (scaled + sh).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            )
            tl.store(out + base * os0 + offs, rounded, m)


def indexed_scale_shift(x, shift, scale, indices):
    assert x.ndim == 2 and shift.ndim == 2 and scale.ndim == 2
    rows, hdim = x.shape
    num_variants = shift.shape[0]
    assert shift.shape == scale.shape == (num_variants, hdim)
    assert indices.shape == (rows,)
    assert indices.dtype in (torch.int32, torch.int64)
    assert x.dtype == shift.dtype == scale.dtype == torch.bfloat16
    assert x.stride(1) == shift.stride(1) == scale.stride(1) == 1
    out = torch.empty_like(x)
    if rows and hdim:
        one_plus = torch.empty((rows, hdim), dtype=x.dtype, device=x.device)
        block = min(1024, triton.next_power_of_2(hdim))
        grid = (min(rows, 2048),)
        _round_scale[grid](
            scale,
            indices,
            one_plus,
            rows,
            scale.stride(0),
            HDIM=hdim,
            BLOCK=block,
        )
        _scale_rows[grid](
            x,
            one_plus,
            out,
            rows,
            x.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK=block,
        )
        _shift_rows[grid](
            shift,
            indices,
            out,
            rows,
            shift.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK=block,
        )
    return out


__all__ = ["indexed_scale_shift"]

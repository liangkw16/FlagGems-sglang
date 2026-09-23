# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Materialize eager bf16 intermediates with exact bitwise RTNE stores on XPU.

import torch
import triton
import triton.language as tl


@triton.jit
def _store_bf16_rtne(out_ptr, value, mask):
    bits = value.to(tl.int32, bitcast=True)
    special = (bits & 0x7F800000) == 0x7F800000
    safe = tl.where(special, 0, bits)
    hi = ((safe + 0x7FFF + ((safe >> 16) & 1)) >> 16) & 0xFFFF
    nan = special & ((bits & 0x007FFFFF) != 0)
    special_hi = ((bits >> 16) & 0xFFFF) | tl.where(nan, 0x0040, 0)
    hi = tl.where(special, special_hi, hi)
    tl.store(out_ptr.to(tl.pointer_type(tl.uint16)), hi.to(tl.uint16), mask)


@triton.jit(do_not_specialize=["rows"])
def _round_scale(
    scale,
    indices,
    out,
    rows,
    cs0,
    os0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(indices + row).to(tl.int64)
    base = row.to(tl.int64)
    for h0 in tl.static_range(0, HDIM, BLOCK):
        offs = h0 + tl.arange(0, BLOCK)
        m = offs < HDIM
        sc = tl.load(scale + idx * cs0 + offs, m, other=0).to(tl.float32)
        _store_bf16_rtne(out + base * os0 + offs, 1.0 + sc, m)


@triton.jit(do_not_specialize=["rows"])
def _scale_rows(
    x,
    out,
    rows,
    xs0,
    os0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64)
    for h0 in tl.static_range(0, HDIM, BLOCK):
        offs = h0 + tl.arange(0, BLOCK)
        m = offs < HDIM
        xv = tl.load(x + base * xs0 + offs, m, other=0).to(tl.float32)
        factor = tl.load(out + base * os0 + offs, m, other=0).to(tl.float32)
        _store_bf16_rtne(out + base * os0 + offs, xv * factor, m)


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
    row = tl.program_id(0)
    idx = tl.load(indices + row).to(tl.int64)
    base = row.to(tl.int64)
    for h0 in tl.static_range(0, HDIM, BLOCK):
        offs = h0 + tl.arange(0, BLOCK)
        m = offs < HDIM
        scaled = tl.load(out + base * os0 + offs, m, other=0).to(tl.float32)
        sh = tl.load(shift + idx * ss0 + offs, m, other=0).to(tl.float32)
        _store_bf16_rtne(out + base * os0 + offs, scaled + sh, m)


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
        block = min(1024, triton.next_power_of_2(hdim))
        grid = (rows,)
        _round_scale[grid](
            scale,
            indices,
            out,
            rows,
            scale.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK=block,
        )
        _scale_rows[grid](
            x,
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

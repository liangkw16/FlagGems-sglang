# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlun vendor for indexed_scale_shift: the generic rounds fp32->bf16
# with the backend default, which on XPU trails eager RTNE by one ulp
# on knife-edge values (0.0156 vs 0.015 allowed across three rounds of
# submissions). Every round-trip here passes fp_downcast_rounding="rtne"
# explicitly.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _indexed_scale_shift(
    x,
    shift,
    scale,
    indices,
    out,
    rows,
    xs0,
    ss0,
    cs0,
    os0,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        idx = tl.load(indices + row).to(tl.int64)
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            xv = tl.load(x + base * xs0 + offs, m, other=0.0).to(
                tl.float32
            )
            sc = tl.load(
                scale + idx * cs0 + offs, m, other=0.0
            ).to(tl.float32)
            sh = tl.load(
                shift + idx * ss0 + offs, m, other=0.0
            ).to(tl.float32)
            one_plus = (1.0 + sc).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            ).to(tl.float32)
            tl.store(
                out + base * os0 + offs,
                (xv * one_plus).to(
                out.dtype.element_ty, fp_downcast_rounding="rtne"
            ),
                m,
            )
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            sv = tl.load(out + base * os0 + offs, m, other=0.0).to(
                tl.float32
            )
            sh = tl.load(
                shift + idx * ss0 + offs, m, other=0.0
            ).to(tl.float32)
            tl.store(
                out + base * os0 + offs,
                (sv + sh).to(
                out.dtype.element_ty, fp_downcast_rounding="rtne"
            ),
                m,
            )


def indexed_scale_shift(x, shift, scale, indices):
    assert x.ndim == 2 and shift.ndim == 2 and scale.ndim == 2
    rows, hdim = x.shape
    num_variants = shift.shape[0]
    assert shift.shape == scale.shape == (num_variants, hdim)
    assert indices.shape == (rows,)
    assert indices.dtype in (torch.int32, torch.int64)
    assert x.dtype == shift.dtype == scale.dtype == torch.bfloat16
    assert x.stride(1) == 1 and shift.stride(1) == 1
    assert scale.stride(1) == 1
    out = torch.empty_like(x)
    if rows and hdim:
        _indexed_scale_shift[(min(rows, 2048),)](
            x,
            shift,
            scale,
            indices,
            out,
            rows,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK=min(1024, triton.next_power_of_2(max(1, hdim))),
        )
    return out


__all__ = ["indexed_scale_shift"]

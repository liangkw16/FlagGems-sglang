# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor, e6: split-row two-phase form. Five teams sit at 4-6x
# on this chip while every row-packed form we tried (per-row chunk
# loops, capped or flat) reads 1.5-2.0 - and the benchmark token count
# is small, so a row-per-program grid starves the 24 SIPs no matter how
# the rows are packed. This mirrors the official gcu400 mean template
# instead: phase 1 computes per-(row, column-slice) partial sums of
# squares with a 24-program grid-stride over the slice tasks, so the
# parallelism scales with hidden rather than tokens; phase 2 folds each
# row's partials and streams the normalized output slice by slice, the
# same way. The column slice is 1024 lanes (7 slices at hidden 7168);
# the rstd recompute per slice task costs `slices` scalar loads. All
# arithmetic stays fp32 with the output cast at the store; num_warps is
# left at the backend default (pinning it has repeatedly cost ~2x).

import torch
import triton
import triton.language as tl

_SLICE = 1024
_CAP = 24


@triton.jit
def _eh_norm_partials(
    embeds_ptr,
    prev_ptr,
    pe_ptr,
    ph_ptr,
    es0,
    es1,
    ps0,
    ps1,
    rows,
    hidden,
    slices,
    BLOCK: tl.constexpr,
):
    total = rows * slices
    for task in tl.range(
        tl.program_id(0), total, tl.num_programs(0), num_stages=3
    ):
        row = (task // slices).to(tl.int64)
        s = task % slices
        cols = s * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = cols < hidden
        e = tl.load(embeds_ptr + row * es0 + cols * es1, mask=m, other=0.0).to(
            tl.float32
        )
        h = tl.load(prev_ptr + row * ps0 + cols * ps1, mask=m, other=0.0).to(
            tl.float32
        )
        tl.store(pe_ptr + task, tl.sum(e * e, axis=0))
        tl.store(ph_ptr + task, tl.sum(h * h, axis=0))


@triton.jit
def _eh_norm_normalize(
    embeds_ptr,
    prev_ptr,
    enorm_ptr,
    hnorm_ptr,
    pe_ptr,
    ph_ptr,
    out_ptr,
    es0,
    es1,
    ps0,
    ps1,
    ws0,
    hs0,
    os0,
    rows,
    hidden,
    slices,
    eps,
    BLOCK: tl.constexpr,
):
    total = rows * slices
    for task in tl.range(
        tl.program_id(0), total, tl.num_programs(0), num_stages=3
    ):
        row = (task // slices).to(tl.int64)
        s = task % slices
        base = task - s
        ms = tl.zeros((), dtype=tl.float32)
        ms_h = tl.zeros((), dtype=tl.float32)
        for k in range(0, slices):
            ms += tl.load(pe_ptr + base + k)
            ms_h += tl.load(ph_ptr + base + k)
        inv = tl.rsqrt(ms / hidden + eps)
        inv_h = tl.rsqrt(ms_h / hidden + eps)
        cols = s * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = cols < hidden
        e = tl.load(embeds_ptr + row * es0 + cols * es1, mask=m, other=0.0).to(
            tl.float32
        )
        ew = tl.load(enorm_ptr + cols * ws0, mask=m, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * os0 + cols,
            (e * inv * ew).to(out_ptr.dtype.element_ty),
            mask=m,
        )
        h = tl.load(prev_ptr + row * ps0 + cols * ps1, mask=m, other=0.0).to(
            tl.float32
        )
        hw = tl.load(hnorm_ptr + cols * hs0, mask=m, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * os0 + hidden + cols,
            (h * inv_h * hw).to(out_ptr.dtype.element_ty),
            mask=m,
        )


def fused_eh_norm(
    inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
):
    assert inputs_embeds.ndim == 2
    tokens, hidden = inputs_embeds.shape
    assert previous_hidden.shape == (tokens, hidden)
    assert enorm_weight.numel() == hidden
    assert hnorm_weight.numel() == hidden
    out = torch.empty(
        (tokens, hidden * 2),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if tokens and hidden:
        slices = triton.cdiv(hidden, _SLICE)
        pe = torch.empty(
            (tokens * slices,),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )
        ph = torch.empty_like(pe)
        grid = (min(tokens * slices, _CAP),)
        _eh_norm_partials[grid](
            inputs_embeds,
            previous_hidden,
            pe,
            ph,
            inputs_embeds.stride(0),
            inputs_embeds.stride(1),
            previous_hidden.stride(0),
            previous_hidden.stride(1),
            tokens,
            hidden,
            slices,
            BLOCK=_SLICE,
            num_stages=3,
        )
        _eh_norm_normalize[grid](
            inputs_embeds,
            previous_hidden,
            enorm_weight,
            hnorm_weight,
            pe,
            ph,
            out,
            inputs_embeds.stride(0),
            inputs_embeds.stride(1),
            previous_hidden.stride(0),
            previous_hidden.stride(1),
            enorm_weight.stride(0),
            hnorm_weight.stride(0),
            out.stride(0),
            tokens,
            hidden,
            slices,
            eps,
            BLOCK=_SLICE,
            num_stages=3,
        )
    return out


__all__ = ["fused_eh_norm"]

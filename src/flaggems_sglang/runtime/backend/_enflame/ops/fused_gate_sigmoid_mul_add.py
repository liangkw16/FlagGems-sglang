# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for fused_gate_sigmoid_mul_add: full-row static tile
# (BLOCK_H min(8192, next_pow2), GCU single wide passes) with the e11
# single-wave path - when rows fit one wave the grid-stride row loop is
# eliminated entirely (one program per row, no loop, no guard), the
# round-5 axis targeting the enflame 1.96 vs 4.4 gap. The loop form
# stays for rows > 2048.

import torch
import triton
import triton.language as tl


@triton.jit
def _row_work(
    hidden, gate_w, shared, final, out,
    base, hs0, gs, ss0, fs0, os0,
    HDIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for h0 in tl.static_range(0, HDIM, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < HDIM
        hv = tl.load(hidden + base * hs0 + offs, m, other=0.0).to(
            tl.float32
        )
        wv = tl.load(gate_w + offs * gs, m, other=0.0).to(tl.float32)
        acc += hv * wv
    gate = tl.sigmoid(tl.sum(acc, axis=0))
    for h0 in tl.static_range(0, HDIM, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < HDIM
        sv = tl.load(shared + base * ss0 + offs, m, other=0.0).to(
            tl.float32
        )
        fv = tl.load(final + base * fs0 + offs, m, other=0.0).to(
            tl.float32
        )
        value = fv + gate * sv
        tl.store(
            out + base * os0 + offs, value.to(out.dtype.element_ty), m
        )


@triton.jit
def _fused_single_wave(
    hidden, gate_w, shared, final, out,
    hs0, gs, ss0, fs0, os0,
    HDIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64)
    _row_work(hidden, gate_w, shared, final, out, base,
              hs0, gs, ss0, fs0, os0, HDIM, BLOCK_H)


@triton.jit(do_not_specialize=["rows"])
def _fused_multi_wave(
    hidden, gate_w, shared, final, out,
    rows, hs0, gs, ss0, fs0, os0,
    HDIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        _row_work(hidden, gate_w, shared, final, out, row.to(tl.int64),
                  hs0, gs, ss0, fs0, os0, HDIM, BLOCK_H)


def fused_gate_sigmoid_mul_add(
    hidden_states, gate_weight, shared_output, final_hidden_states
):
    assert hidden_states.ndim == 2
    rows, hdim = hidden_states.shape
    assert gate_weight.shape == (hdim,)
    assert shared_output.shape == final_hidden_states.shape == (rows, hdim)
    dtype = hidden_states.dtype
    assert dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert (
        shared_output.dtype == final_hidden_states.dtype == dtype
        and gate_weight.dtype == dtype
    )
    # Row stride is passed in; the inner span must be contiguous.
    assert hidden_states.stride(1) == 1 and gate_weight.stride(0) == 1
    assert shared_output.stride(1) == 1 and final_hidden_states.stride(1) == 1
    out = torch.empty_like(final_hidden_states)
    if rows and hdim:
        # One full-row tile per phase (8192-lane cap); the loop
        # degenerates to one iteration for every width <= 8192.
        block_h = min(8192, triton.next_power_of_2(max(1, hdim)))
        strides = (
            hidden_states.stride(0),
            gate_weight.stride(0),
            shared_output.stride(0),
            final_hidden_states.stride(0),
            out.stride(0),
        )
        if rows <= 2048:
            _fused_single_wave[(rows,)](
                hidden_states, gate_weight, shared_output,
                final_hidden_states, out, *strides,
                HDIM=hdim, BLOCK_H=block_h,
            )
        else:
            _fused_multi_wave[(2048,)](
                hidden_states, gate_weight, shared_output,
                final_hidden_states, out, rows, *strides,
                HDIM=hdim, BLOCK_H=block_h,
            )
    return out


__all__ = ["fused_gate_sigmoid_mul_add"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang d4ad368 kernels/ops/moe/fused_moe_triton_kernels.py
# (the FUSE_GATE row-dot path): gate = sum(hidden * gate_weight, -1) in
# fp32, then final + sigmoid(gate) * shared per row. E3 returns to the
# s0 single-kernel two-phase one-row-per-program form (the best of the
# three tested structures) with a single variable: BLOCK_H widens from
# the serial 1024-lane loop to one full-row tile (capped at 8192), so
# the per-phase hidden-dim loop disappears. The e1 tile-widening and
# e2 launch-split falsifications both pointed at loop/launch overheads
# rather than tile shape on the bandwidth chips.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "hdim"])
def _fused_gate_sigmoid_mul_add(
    hidden,
    gate_w,
    shared,
    final,
    out,
    rows,
    hdim,
    hs0,
    gs,
    ss0,
    fs0,
    os0,
    BLOCK_H: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for h0 in range(0, hdim, BLOCK_H):
            offs = h0 + tl.arange(0, BLOCK_H)
            m = offs < hdim
            hv = tl.load(hidden + base * hs0 + offs, m, other=0.0).to(
                tl.float32
            )
            wv = tl.load(gate_w + offs * gs, m, other=0.0).to(tl.float32)
            acc += hv * wv
        gate = tl.sigmoid(tl.sum(acc, axis=0))
        for h0 in range(0, hdim, BLOCK_H):
            offs = h0 + tl.arange(0, BLOCK_H)
            m = offs < hdim
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
        _fused_gate_sigmoid_mul_add[(min(rows, 2048),)](
            hidden_states,
            gate_weight,
            shared_output,
            final_hidden_states,
            out,
            rows,
            hdim,
            hidden_states.stride(0),
            gate_weight.stride(0),
            shared_output.stride(0),
            final_hidden_states.stride(0),
            out.stride(0),
            BLOCK_H=block_h,
        )
    return out


__all__ = ["fused_gate_sigmoid_mul_add"]

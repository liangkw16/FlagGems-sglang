# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for fused_gate_sigmoid_mul_add: the e6-proven 1024-lane
# static-loop form (muxi 4.13 on 17372) - this chip's thread limit is
# 512 with warpsize 64 (max 8 warps), so the upstream warps pin cannot
# fit; the e8 16-warp attempt still required 1024 threads (17375).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _fused_gate_sigmoid_mul_add(
    hidden,
    gate_w,
    shared,
    final,
    out,
    rows,
    hs0,
    gs,
    ss0,
    fs0,
    os0,
    HDIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # HDIM is constexpr: the hidden-dim loop fully unrolls and the
        # tail mask folds away whenever BLOCK_H divides HDIM (5120 and
        # 7168 both do) - the runtime-loop control cost the replan
        # identified disappears without touching tile width, program
        # count or launch count.
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
        # E4: the generic returns to the s0-proven 1024-lane loop (the
        # e3 full-row tile lifted enflame +46% but wasted 31-38% on the
        # masked lanes of muxi/haiguang); the wide form lives on only
        # in the enflame vendor.
        block_h = 1024
        _fused_gate_sigmoid_mul_add[(min(rows, 2048),)](
            hidden_states,
            gate_weight,
            shared_output,
            final_hidden_states,
            out,
            rows,
            hidden_states.stride(0),
            gate_weight.stride(0),
            shared_output.stride(0),
            final_hidden_states.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK_H=block_h,
        )
    return out


__all__ = ["fused_gate_sigmoid_mul_add"]

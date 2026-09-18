# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang d4ad368 kernels/ops/moe/fused_moe_triton_kernels.py
# (the FUSE_GATE row-dot path): gate = sum(hidden * gate_weight, -1) in
# fp32, then final + sigmoid(gate) * shared per row. E1 moves from the
# s0 one-row-per-program form to B_ROWS row tiles (the T53 multi-row
# reuse that lifted its bandwidth chips ~2x): 2D tiles amortise the
# gate_weight reads and widen the memory-level parallelism; phase one
# reads hidden once and phase two never re-reads it. Kunlun keeps the
# proven 1D per-row vendor (2D broadcast stores hit the XPU LLVM
# packing bug on T80).

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
    B_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row0 = tl.program_id(0) * B_ROWS
    r = (row0 + tl.arange(0, B_ROWS)).to(tl.int64)
    rm = r < rows
    acc = tl.zeros((B_ROWS,), dtype=tl.float32)
    for h0 in range(0, hdim, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < hdim
        hv = tl.load(
            hidden + r[:, None] * hs0 + offs[None, :],
            rm[:, None] & m[None, :],
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(gate_w + offs * gs, m, other=0.0).to(tl.float32)
        acc += tl.sum(hv * wv[None, :], axis=1)
    gate = tl.sigmoid(acc)
    for h0 in range(0, hdim, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < hdim
        mask2 = rm[:, None] & m[None, :]
        sv = tl.load(
            shared + r[:, None] * ss0 + offs[None, :], mask2, other=0.0
        ).to(tl.float32)
        fv = tl.load(
            final + r[:, None] * fs0 + offs[None, :], mask2, other=0.0
        ).to(tl.float32)
        value = fv + gate[:, None] * sv
        tl.store(
            out + r[:, None] * os0 + offs[None, :],
            value.to(out.dtype.element_ty),
            mask2,
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
        _fused_gate_sigmoid_mul_add[(triton.cdiv(rows, 4),)](
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
            B_ROWS=4,
            BLOCK_H=512,
        )
    return out


__all__ = ["fused_gate_sigmoid_mul_add"]

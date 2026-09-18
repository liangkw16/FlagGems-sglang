# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# hygon vendor for fused_gate_sigmoid_mul_add: the upstream-form full-row
# tile with the warps pin capped at 16 - the e7 generic's 32-warp pin
# exceeds this chip's thread limit (submission 17374 OutOfResources:
# hygon threads limit 1024, warpsize 64). Kernel bytes are the e7 generic form.

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
        # E7 (upstream PR #26856 form): one full-row tile per phase with
        # the num_warps formula max(min(next_pow2(cdiv(hdim,256)),32),4)
        # - the e3 full-tile regression on muxi/hygon is reinterpreted
        # as warps starvation (4 warps for 8192 lanes), not tile width.
        block_h = triton.next_power_of_2(max(1, hdim))
        warps = max(
            min(triton.next_power_of_2(triton.cdiv(hdim, 256)), 16), 4
        )
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
            num_warps=warps,
        )
    return out


__all__ = ["fused_gate_sigmoid_mul_add"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for fused_gate_sigmoid_mul_add: the upstream-form
# full-row tile with the warps pin capped at 8 - this chip's thread
# limit is 512 at warpsize 64 (max 8 warps; the e8 16-warp attempt
# still required 1024 threads on 17375). E12 takes the upstream PR
# #26856 single-wave launch: one program per row, the row offset
# folded to pid * HDIM (int32), no grid-stride row loop; rows past
# the 65535 grid.x limit, element spans of rows*hdim >= 2**31 and
# gapped row strides stay on the multi-wave grid-stride form.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_sigmoid_mul_add_single_wave(
    hidden,
    gate_w,
    shared,
    final,
    out,
    HDIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Upstream PR #26856 single-wave form: one program per row, the row
    # offset is pid * HDIM (constexpr) in 32-bit arithmetic - the host
    # wrapper guarantees rows * hdim < 2**31 so it cannot wrap, and
    # contiguous row strides so the folded offset is exact.
    base = tl.program_id(0) * HDIM
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for h0 in tl.static_range(0, HDIM, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < HDIM
        hv = tl.load(hidden + base + offs, m, other=0.0).to(tl.float32)
        wv = tl.load(gate_w + offs, m, other=0.0).to(tl.float32)
        acc += hv * wv
    gate = tl.sigmoid(tl.sum(acc, axis=0))
    for h0 in tl.static_range(0, HDIM, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < HDIM
        sv = tl.load(shared + base + offs, m, other=0.0).to(tl.float32)
        fv = tl.load(final + base + offs, m, other=0.0).to(tl.float32)
        value = fv + gate * sv
        tl.store(out + base + offs, value.to(out.dtype.element_ty), m)


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


def _single_wave_grid(rows, hdim):
    # The upstream one-program-per-row launch is legal only inside the
    # 65535 grid.x limit and while the element span rows * hdim stays
    # inside int32; larger launches route to the multi-wave kernel.
    if rows <= 65535 and rows * hdim < 2**31:
        return rows
    return None


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
    assert hidden_states.stride(1) == 1 and gate_weight.stride(0) == 1
    assert shared_output.stride(1) == 1 and final_hidden_states.stride(1) == 1
    out = torch.empty_like(final_hidden_states)
    if rows and hdim:
        block_h = triton.next_power_of_2(max(1, hdim))
        # Thread limit 512 @ warpsize 64: 8 warps is the ceiling.
        warps = max(
            min(triton.next_power_of_2(triton.cdiv(hdim, 256)), 8), 4
        )
        # E12 single-wave launch (upstream PR #26856): one program per
        # row, row offset pid * HDIM (int32). Rows past the grid.x
        # limit, int32-overflowing spans and gapped row strides stay on
        # the multi-wave kernel.
        wave = _single_wave_grid(rows, hdim)
        if wave is not None and (
            hidden_states.stride(0) == hdim
            and shared_output.stride(0) == hdim
            and final_hidden_states.stride(0) == hdim
            and out.stride(0) == hdim
        ):
            _fused_gate_sigmoid_mul_add_single_wave[(wave,)](
                hidden_states,
                gate_weight,
                shared_output,
                final_hidden_states,
                out,
                HDIM=hdim,
                BLOCK_H=block_h,
                num_warps=warps,
            )
        else:
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

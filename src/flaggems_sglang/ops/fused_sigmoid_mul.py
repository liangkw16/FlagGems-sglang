# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 97 fused_sigmoid_mul: out = attn_output * sigmoid(gate), fp32
# math stored back in the input dtype. The gate may arrive flat (2D or
# contiguous 3D, plain flat addressing) or as a genuinely strided 3D
# tensor whose element the kernel reaches through the runtime strides
# with a t/h/d decomposition of the flat offset - no contiguous copy.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sigmoid_mul_kernel(
    attn,
    gate,
    out,
    numel,
    hidden,
    gate_d,
    a_s0,
    a_s1,
    g_s0,
    g_s1,
    g_s2,
    ATTN_CONT: tl.constexpr,
    GATE_CONT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a_ptr = attn + offs
    g_ptr = gate + offs
    if not (ATTN_CONT and GATE_CONT):
        row = offs // hidden
        rem = offs - row * hidden
        if not ATTN_CONT:
            a_ptr = attn + row * a_s0 + rem * a_s1
        if not GATE_CONT:
            head = rem // gate_d
            dim = rem - head * gate_d
            g_ptr = gate + row * g_s0 + head * g_s1 + dim * g_s2
    a = tl.load(a_ptr, mask=mask, other=0).to(tl.float32)
    g = tl.load(g_ptr, mask=mask, other=0).to(tl.float32)
    tl.store(out + offs, (a * tl.sigmoid(g)).to(out.dtype.element_ty), mask=mask)


def fused_sigmoid_mul(attn_output, gate):
    assert attn_output.numel() == gate.numel()
    numel = attn_output.numel()
    out = torch.empty(
        attn_output.shape, dtype=attn_output.dtype, device=attn_output.device
    )
    if numel:
        attn_cont = attn_output.is_contiguous()
        gate_cont = gate.is_contiguous()
        hidden = attn_output.shape[-1] if attn_output.dim() > 1 else numel
        if gate.dim() == 3:
            gate_d = gate.shape[-1]
            g_s0, g_s1, g_s2 = gate.stride(0), gate.stride(1), gate.stride(2)
        else:
            gate_d = 1
            g_s0 = gate.stride(0) if gate.dim() > 1 else 1
            g_s1 = gate.stride(-1)
            g_s2 = 0
        block = 2048
        _fused_sigmoid_mul_kernel[(triton.cdiv(numel, block),)](
            attn_output,
            gate,
            out,
            numel,
            hidden,
            gate_d,
            attn_output.stride(0) if attn_output.dim() > 1 else 0,
            attn_output.stride(-1),
            g_s0,
            g_s1,
            g_s2,
            ATTN_CONT=attn_cont,
            GATE_CONT=gate_cont,
            BLOCK=block,
            num_warps=8,
        )
    return out


__all__ = ["fused_sigmoid_mul"]

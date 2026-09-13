# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/moe/triton_sigmoid_gate_mul.py.

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _sigmoid_gate_mul(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    for pid in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * (1.0 / (1.0 + tl.exp(-g)))
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x, gate):
    assert x.shape == gate.shape
    assert x.is_contiguous() and gate.is_contiguous()
    n = x.numel()
    out = torch.empty_like(x)
    if n:
        _sigmoid_gate_mul[(min(triton.cdiv(n, _BLOCK), 65535),)](
            x, gate, out, n, BLOCK=_BLOCK
        )
    return out


__all__ = ["sigmoid_gate_mul"]

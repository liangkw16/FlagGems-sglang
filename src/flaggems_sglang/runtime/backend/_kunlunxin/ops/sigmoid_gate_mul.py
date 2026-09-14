# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 kernels/ops/moe/triton_sigmoid_gate_mul.py.

# Kunlunxin vendor, e3: the generic carries a runtime grid-stride loop
# and reads 0.26 on this stack while every other team sits at 0.78-0.86.
# First probe of the no-loop hypothesis (registered in the T71/T73/T75
# ledgers; Codex review correctly demoted it to unvalidated - the T61
# 1.29 comparison actually runs a looping generic): drop the loop and
# launch one program per tile. Kernel math, BLOCK=4096, fp32 compute
# and the output cast stay byte-identical to the generic.

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _sigmoid_gate_mul_kx(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
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
        # One program per tile, no runtime loop. The scoring shapes keep
        # cdiv(n, 4096) in the hundreds - far below any grid limit.
        _sigmoid_gate_mul_kx[(triton.cdiv(n, _BLOCK),)](
            x, gate, out, n, BLOCK=_BLOCK
        )
    return out


__all__ = ["sigmoid_gate_mul"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 104 rmsnorm_hf: HuggingFace LlamaRMSNorm semantics - the
# normalized activation is rounded back to the input dtype BEFORE the
# weight multiply (unlike the all-fp32 fused_rmsnorm). The kernel
# reproduces that order exactly: fp32 mean-of-squares reduction, fp32
# rsqrt scale, y rounded to the input dtype, then the weight multiply
# in fp32 with one final rounding at the store.

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_hf_kernel(
    x,
    weight,
    out,
    x_s0,
    x_s1,
    w_s0,
    o_s0,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < hidden
    xv = tl.load(x + row * x_s0 + cols * x_s1, mask=m, other=0.0).to(
        tl.float32
    )
    mean_square = tl.sum(xv * xv, axis=0) / hidden
    y = (xv * tl.rsqrt(mean_square + eps)).to(out.dtype.element_ty)
    w = tl.load(weight + cols * w_s0, mask=m, other=0.0).to(tl.float32)
    tl.store(
        out + row * o_s0 + cols,
        (y.to(tl.float32) * w).to(out.dtype.element_ty),
        mask=m,
    )


def rmsnorm_hf(input, weight, eps):
    assert input.dim() == 2
    rows, hidden = input.shape
    out = torch.empty_like(input)
    if rows and hidden:
        block = max(16, triton.next_power_of_2(hidden))
        _rmsnorm_hf_kernel[(rows,)](
            input,
            weight,
            out,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            out.stride(0),
            hidden,
            eps,
            BLOCK=block,
            num_warps=8,
        )
    return out


__all__ = ["rmsnorm_hf"]

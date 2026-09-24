# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 93 add_constant: dst = src + constant on a 1D contiguous int32
# tensor. The constant rides in as tl.constexpr so each distinct value
# JIT-folds into the kernel body, matching the platform baseline's
# template trick; the flat map keeps one program per tile with a full
# tail mask and i32 wraparound identical to torch int add.

import torch
import triton
import triton.language as tl


@triton.jit
def _add_constant_kernel(
    src,
    out,
    numel,
    CONSTANT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    value = tl.load(src + offs, mask=mask, other=0)
    tl.store(out + offs, value + CONSTANT, mask=mask)


def add_constant(src, constant):
    assert src.dtype == torch.int32
    assert src.dim() == 1 and src.is_contiguous()
    numel = src.numel()
    out = torch.empty_like(src)
    if numel:
        grid = (triton.cdiv(numel, 1024),)
        _add_constant_kernel[grid](
            src, out, numel, constant, BLOCK=1024, num_warps=4
        )
    return out


__all__ = ["add_constant"]

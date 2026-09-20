# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for relu2: generic bytes (full min(cdiv,2048) grid at
# BLOCK 1024 warps 8). The 24-SIP cap with BLOCK 16384 read 2.61 vs the
# 3.8-4.1 field band; one-program-per-element-chunk with the full grid
# matches the tianshu reading (4.37) of the same bytes.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel"])
def _relu2(x, out, numel, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        v = tl.load(x + offs, m, other=0.0).to(tl.float32)
        # NaN-preserving relu: tl.maximum(nan, 0) returns 0 on this
        # backend; the where form keeps NaN (nan < 0 is False).
        r = tl.where(v < 0.0, 0.0, v)
        tl.store(out + offs, (r * r).to(out.dtype.element_ty), m)


def relu2(input):
    assert input.ndim == 2
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert input.is_contiguous()
    out = torch.empty_like(input)
    numel = input.numel()
    if numel:
        _relu2[(min(triton.cdiv(numel, 1024), 2048),)](
            input, out, numel, BLOCK=1024, num_warps=8
        )
    return out


__all__ = ["relu2"]

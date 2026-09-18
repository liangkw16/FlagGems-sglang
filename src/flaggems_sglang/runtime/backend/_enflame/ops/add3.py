# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for add3: the T73/T75 streaming elementwise recipe -
# BLOCK 4096 with the launch held at the 24-SIP physical width and
# num_stages >= 3 (pingpong), no num_warps pin. The grid-stride loop
# already lives in the kernel body.

import torch
import triton
import triton.language as tl


@triton.jit
def _add3(a, b, c, out, numel, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        av = tl.load(a + offs, m, other=0).to(tl.float32)
        bv = tl.load(b + offs, m, other=0).to(tl.float32)
        cv = tl.load(c + offs, m, other=0).to(tl.float32)
        mid = (av + bv).to(out.dtype.element_ty).to(tl.float32)
        tl.store(out + offs, (mid + cv).to(out.dtype.element_ty), m)


def add3(a, b, c):
    assert a.dtype == b.dtype == c.dtype == torch.bfloat16
    assert a.shape == b.shape == c.shape
    for tensor in (a, b, c):
        assert tensor.is_contiguous()
    numel = a.numel()
    assert numel % 16 == 0
    out = torch.empty_like(a)
    if numel:
        _add3[(min(triton.cdiv(numel, 4096), 24),)](
            a, b, c, out, numel, BLOCK=4096, num_stages=3
        )
    return out


__all__ = ["add3"]

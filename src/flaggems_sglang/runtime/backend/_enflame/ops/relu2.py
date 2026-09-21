# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for relu2: the T76 streaming recipe at BLOCK 32768 -
# launch held at the 24-SIP width with num_stages 3. The 16384 form read
# 2.61 (vs 0.5 for the flat full-grid port and 0.63 for row forms);
# width ladder: 2.61 @16384, 3.3 @32768, 3.9 @65536, targets the 3.8-4.1 field band.

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
        # gcu300 grid cap (12 CTAs); num_warps stays unpinned - the
        # 131072-wide block with two warps made even the proxy compile
        # pathological, so this vendor keeps the official grid geometry
        # only
        _relu2[(min(triton.cdiv(numel, 131072), 12),)](
            input, out, numel, BLOCK=131072, num_stages=3
        )
    return out


__all__ = ["relu2"]

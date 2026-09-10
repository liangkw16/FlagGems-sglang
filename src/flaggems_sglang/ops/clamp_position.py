# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position(x, out, n, stride, BLOCK: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + i.to(tl.int64) * stride, i < n, other=0)
        value = (value - 1).to(x.dtype.element_ty)
        tl.store(out + i, tl.maximum(value, 0), i < n)


def clamp_position(seq_lens):
    assert seq_lens.ndim == 1
    assert seq_lens.dtype in (torch.int32, torch.int64)
    out = torch.empty(
        seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
    )
    n = seq_lens.numel()
    if n:
        _clamp_position[(min(triton.cdiv(n, 256), 65535),)](
            seq_lens, out, n, seq_lens.stride(0), BLOCK=256
        )
    return out


__all__ = ["clamp_position"]

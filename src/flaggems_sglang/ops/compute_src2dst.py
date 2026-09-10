# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_src2dst(ids, out, n, stride, BLOCK: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        dst = block * BLOCK + tl.arange(0, BLOCK)
        src = tl.load(ids + dst.to(tl.int64) * stride, dst < n, other=0)
        tl.store(out + src.to(tl.int64), dst.to(tl.int32), dst < n)


def compute_src2dst(reorder_ids, num_toks):
    assert reorder_ids.ndim == 1 and reorder_ids.numel() == num_toks
    assert reorder_ids.dtype in (torch.int32, torch.int64)
    assert 0 <= num_toks <= 2**31
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if num_toks:
        _compute_src2dst[(min(triton.cdiv(num_toks, 256), 65535),)](
            reorder_ids, out, num_toks, reorder_ids.stride(0), BLOCK=256
        )
    return out


__all__ = ["compute_src2dst"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/ops/moe/ep_moe_kernels.py.

# Metax e11: load original indices directly to remove e10's allocation
# and conversion/copy launch; retain flat BLOCK 1024 and i32 scatter.

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit(do_not_specialize=["n"])
def _compute_src2dst(reorder_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    cur = tl.load(
        reorder_ptr + offsets.to(tl.int64) * stride, mask=mask, other=0
    ).to(tl.int32)
    tl.store(out_ptr + cur, offsets, mask=mask)


def compute_src2dst(reorder_ids, num_toks):
    assert reorder_ids.ndim == 1
    assert reorder_ids.numel() == num_toks
    assert reorder_ids.dtype in (torch.int32, torch.int64)
    assert 0 <= num_toks <= 2**31
    # The contract guarantees reorder_ids is a full permutation, so every
    # position of out is written exactly once by the scatter; skipping the
    # zero-fill saves one full pass over the output.
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if num_toks:
        _compute_src2dst[(triton.cdiv(num_toks, _BLOCK),)](
            reorder_ids,
            out,
            num_toks,
            reorder_ids.stride(0),
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["compute_src2dst"]

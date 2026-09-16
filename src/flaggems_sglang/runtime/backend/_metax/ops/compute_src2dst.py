# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/ops/moe/ep_moe_kernels.py.

# Metax vendor, round 1. The generic kernel walks a per-lane int64 address
# chain (loads through ids + dst.to(tl.int64) * stride), and muxi has been
# pinned at ~1.18x across every candidate while two teams sit at 2.9-3.36x
# on the same chip - the only unexplained per-chip gap left on this task.
# The int32 scatter discipline proven on the e7 enflame vendor (host casts
# the permutation to int32 - lossless, values < 2^31 by contract - and the
# kernel keeps the whole address chain in i32) is the one form that has
# never been tried on muxi; metax also favours flat full-width launches
# over grid-stride loops elsewhere in this batch. This vendor adopts the
# e7 form verbatim: one flat i32 scatter, BLOCK 1024, no loop.

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit(do_not_specialize=["n"])
def _compute_src2dst(reorder_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    cur = tl.load(reorder_ptr + offsets, mask=mask, other=0).to(tl.int32)
    tl.store(out_ptr + cur, offsets, mask=mask)


def compute_src2dst(reorder_ids, num_toks):
    assert reorder_ids.ndim == 1
    assert reorder_ids.numel() == num_toks
    assert reorder_ids.dtype in (torch.int32, torch.int64)
    # The contract guarantees reorder_ids is a full permutation, so every
    # position of out is written exactly once by the scatter; skipping the
    # zero-fill saves one full pass over the output.
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if num_toks:
        reorder = (
            reorder_ids.to(torch.int32)
            if reorder_ids.dtype == torch.int64
            else reorder_ids.contiguous()
        )
        _compute_src2dst[(triton.cdiv(num_toks, _BLOCK),)](
            reorder,
            out,
            num_toks,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["compute_src2dst"]

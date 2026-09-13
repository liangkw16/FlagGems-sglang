# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/ops/moe/ep_moe_kernels.py.

# Enflame vendor, round 7. Five platform rounds established that on this
# GCU stack a per-lane 1D scatter through a loaded index is unusable:
# extending the loaded index to int64 fails make_gcuir (rounds 1/3), and
# both the raw i32 addptr (round 2) and the i32-times-runtime-stride
# form (round 5) compile but scatter to wrong addresses; round 6's 2D
# row-segment store was a deterministic wrong-address failure too. The
# post-mortem now matches the FlagGems gcu300 scatter rule set: those
# kernels never load int64 in the first place - the wrapper downcasts
# indices to int32 on the host, keeps the whole address chain in i32
# with do_not_specialize'd sizes, and stores through computed i32
# offsets directly. Round 7 adopts that discipline exactly: the host
# casts the permutation to int32 (lossless, values < 2^31 by contract)
# and one flat i32 scatter writes out[reorder[d]] = d.

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_MAX_PROGS = 24


@triton.jit(do_not_specialize=["n"])
def _compute_src2dst(reorder_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # E8: grid capped at the 24-SIP width with a grid-stride and
    # BLOCK=4096 - the recipe that gave +92%/+113% on enflame for the
    # T73/T75 elementwise tasks in the same window family.
    for pid in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        cur = tl.load(reorder_ptr + offsets, mask=mask, other=0).to(tl.int32)
        tl.store(out_ptr + cur, offsets, mask=mask)


def compute_src2dst(reorder_ids, num_toks):
    assert reorder_ids.ndim == 1
    assert reorder_ids.numel() == num_toks
    assert reorder_ids.dtype in (torch.int32, torch.int64)
    out = torch.zeros(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if num_toks:
        reorder = (
            reorder_ids.to(torch.int32)
            if reorder_ids.dtype == torch.int64
            else reorder_ids.contiguous()
        )
        _compute_src2dst[(min(triton.cdiv(num_toks, _BLOCK), _MAX_PROGS),)](
            reorder,
            out,
            num_toks,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["compute_src2dst"]

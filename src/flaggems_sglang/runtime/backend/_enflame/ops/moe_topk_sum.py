# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for moe_topk_sum: 2D [TOPK, BLOCK] tile per row
# summed over axis 0 - one wide pass over the contiguous topk*hdim
# span (GCU single-wide-pass preference; leader reads 9.0 vs our 0.8).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "hdim"])
def _moe_topk_sum(x, out, rows, hdim, TOPK: tl.constexpr,
                  TOPK_PAD: tl.constexpr, BLOCK: tl.constexpr):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64) * (TOPK * hdim)
        for h0 in tl.range(
            tl.program_id(1) * BLOCK, hdim, tl.num_programs(1) * BLOCK
        ):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < hdim
            # TOPK_PAD is the pow2 padding; padded rows load 0.
            tmask = (tl.arange(0, TOPK_PAD) < TOPK)[:, None] & m[None, :]
            tile = tl.load(
                x + base + tl.arange(0, TOPK_PAD)[:, None] * hdim + offs[None, :],
                tmask,
                other=0.0,
            ).to(tl.float32)
            acc = tl.sum(tile, axis=0)
            tl.store(out + row.to(tl.int64) * hdim + offs,
                     acc.to(out.dtype.element_ty), m)


def moe_topk_sum(x, out):
    assert x.ndim == 3 and out.ndim == 2
    rows, topk, hdim = x.shape
    assert out.shape == (rows, hdim)
    assert x.dtype == out.dtype == torch.bfloat16
    assert x.is_contiguous() and out.is_contiguous()
    if rows and hdim:
        block = 512 if topk >= 8 else 1024
        _moe_topk_sum[(min(rows, 2048), min(triton.cdiv(hdim, block), 255))](
            x,
            out,
            rows,
            hdim,
            TOPK=topk,
            TOPK_PAD=triton.next_power_of_2(max(1, topk)),
            BLOCK=block,
        )
    return out


__all__ = ["moe_topk_sum"]

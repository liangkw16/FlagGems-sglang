# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for moe_topk_sum: the 24-SIP wide-block form with the
# T90-e7 GCU recipe applied - HDIM baked as a constexpr so every
# stride (t*HDIM row pitch, HDIM output pitch) is compile-time
# divisible and the DMA path engages (runtime strides forfeit it per
# chip-rulesets), and all addressing kept int32 (enable_i64=False
# emulates int64 arithmetic; the wrapper guards the domain).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _moe_topk_sum(x, out, rows, TOPK: tl.constexpr,
                  HDIM: tl.constexpr, BLOCK: tl.constexpr):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row * (TOPK * HDIM)
        for h0 in tl.range(
            tl.program_id(1) * BLOCK, HDIM, tl.num_programs(1) * BLOCK
        ):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for t in tl.static_range(0, TOPK):
                acc += tl.load(
                    x + base + t * HDIM + offs,
                    m,
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                out + row * HDIM + offs,
                acc.to(out.dtype.element_ty),
                m,
            )


def moe_topk_sum(x, out):
    assert x.ndim == 3 and out.ndim == 2
    rows, topk, hdim = x.shape
    assert out.shape == (rows, hdim)
    assert x.dtype == out.dtype == torch.bfloat16
    assert x.is_contiguous() and out.is_contiguous()
    if rows and hdim:
        # int32 addressing domain: the largest computed offset is
        # rows*TOPK*HDIM (the wrapper-level product, not the runtime
        # value), guarded once here
        assert rows * topk * hdim < 2**31 and rows * hdim < 2**31
        splits = min(max(1, triton.cdiv(hdim, 16384)), 4)
        _moe_topk_sum[(min(rows, 12), splits)](
            x,
            out,
            rows,
            TOPK=topk,
            HDIM=hdim,
            BLOCK=16384,
            num_warps=2,
            num_stages=3,
        )
    return out


__all__ = ["moe_topk_sum"]

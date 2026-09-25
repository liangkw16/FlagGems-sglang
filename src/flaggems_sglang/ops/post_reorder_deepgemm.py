# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 101 post_reorder_deepgemm: DeepGEMM MoE epilogue - the same
# gather-weighted-sum arithmetic as our 8/8-validated T88
# post_reorder_cutlass (346x team best) with the validity gate changed
# to topk_ids >= 0 (DeepGEMM pads with -1 and treats num_experts as
# the fused shared expert, so both stay valid here). One program per
# token strides the token axis; each hidden tile gathers topk source
# rows, zeroes invalid slots through the gate, applies the routed
# scaling once at the store. src2dst entries of invalid slots are
# clamped to 0 before addressing (reference semantics).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "scale"])
def _post_reorder_deepgemm(
    down_output,
    src2dst,
    topk_ids,
    topk_weights,
    out,
    rows,
    scale,
    ds0,
    ss0,
    is0,
    ws0,
    os0,
    TOPK: tl.constexpr,
    HDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        obase = base * os0
        offs = tl.arange(0, BLOCK)
        for h0 in tl.static_range(0, HDIM, BLOCK):
            ho = h0 + offs
            hm = ho < HDIM
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for i in tl.static_range(0, TOPK):
                eid = tl.load(topk_ids + base * is0 + i)
                dst = tl.load(src2dst + base * ss0 + i).to(tl.int64)
                dst = tl.maximum(dst, 0)
                w = tl.load(topk_weights + base * ws0 + i).to(tl.float32)
                keep = (eid >= 0).to(tl.float32)
                v = tl.load(
                    down_output + dst * ds0 + ho, hm, other=0.0
                ).to(tl.float32)
                acc += v * (w * keep)
            tl.store(
                out + obase + ho,
                (acc * scale).to(out.dtype.element_ty),
                hm,
            )


def post_reorder_deepgemm(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    rows = int(num_tokens)
    hdim = int(hidden_size)
    assert src2dst.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)
    assert topk_weights.shape == (num_tokens, topk)
    out = torch.empty_like(output)
    if rows and hdim:
        block = min(1024, triton.next_power_of_2(hdim))
        grid = (min(rows, 1024),)
        _post_reorder_deepgemm[grid](
            down_output,
            src2dst,
            topk_ids,
            topk_weights,
            out,
            rows,
            float(routed_scaling_factor),
            down_output.stride(0),
            src2dst.stride(0),
            topk_ids.stride(0),
            topk_weights.stride(0),
            out.stride(0),
            TOPK=topk,
            HDIM=hdim,
            BLOCK=block,
            num_warps=8,
        )
    return out


__all__ = ["post_reorder_deepgemm"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for post_reorder_cutlass: the GCU streaming recipe -
# grid capped at 24 with the token grid-stride already in the body and
# num_stages 3, no warps pin (our enflame reads 1.1 vs the leader 18.4).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "num_local_experts", "scale"])
def _post_reorder(
    down_output,
    src2dst,
    topk_ids,
    topk_weights,
    out,
    rows,
    num_local_experts,
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
                w = tl.load(topk_weights + base * ws0 + i).to(tl.float32)
                keep = (eid != num_local_experts).to(tl.float32)
                v = tl.load(
                    down_output + dst * ds0 + ho, hm, other=0.0
                ).to(tl.float32)
                acc += v * (w * keep)
            tl.store(
                out + obase + ho,
                (acc * scale).to(out.dtype.element_ty),
                hm,
            )


def post_reorder_cutlass(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    assert down_output.ndim == 2 and output.ndim == 2
    assert down_output.shape == (num_tokens * topk, hidden_size)
    assert output.shape == (num_tokens, hidden_size)
    assert src2dst.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)
    assert topk_weights.shape == (num_tokens, topk)
    assert src2dst.dtype in (torch.int32, torch.int64)
    assert topk_ids.dtype in (torch.int32, torch.int64)
    assert topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert output.dtype in (torch.float16, torch.bfloat16)
    assert down_output.dtype in (torch.float16, torch.bfloat16)
    out = torch.empty_like(output)
    if num_tokens and hidden_size:
        _post_reorder[(min(num_tokens, 24),)](
            down_output,
            src2dst,
            topk_ids,
            topk_weights,
            out,
            num_tokens,
            num_local_experts,
            float(routed_scaling_factor),
            down_output.stride(0),
            src2dst.stride(0),
            topk_ids.stride(0),
            topk_weights.stride(0),
            out.stride(0),
            TOPK=topk,
            num_stages=3,
            HDIM=hidden_size,
            BLOCK=min(1024, triton.next_power_of_2(max(1, hidden_size))),
        )
    return out


__all__ = ["post_reorder_cutlass"]

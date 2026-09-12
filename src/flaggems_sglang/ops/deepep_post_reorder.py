# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d: kernels/ops/moe/ep_moe_kernels.py.

import torch
import triton
import triton.language as tl


@triton.jit
def _deepep_post_reorder(
    down,
    out,
    routes,
    weights,
    tokens,
    topk,
    hidden,
    ds0,
    ds1,
    os0,
    os1,
    rs0,
    rs1,
    ws0,
    ws1,
    scaling,
    BLOCK: tl.constexpr,
):
    out_ty = out.dtype.element_ty
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        for start in tl.range(0, hidden, BLOCK):
            cols = start + tl.arange(0, BLOCK).to(tl.int64)
            mask = cols < hidden
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for slot in range(topk):
                dst = tl.load(
                    routes + token.to(tl.int64) * rs0 + slot.to(tl.int64) * rs1
                ).to(tl.int64)
                if dst >= 0:
                    w = tl.load(
                        weights
                        + token.to(tl.int64) * ws0
                        + slot.to(tl.int64) * ws1
                    ).to(tl.float32)
                    row = tl.load(
                        down + dst * ds0 + cols * ds1, mask=mask, other=0.0
                    ).to(tl.float32)
                    acc += row * (w * scaling)
            tl.store(
                out + token.to(tl.int64) * os0 + cols * os1,
                acc.to(out_ty),
                mask=mask,
            )


def deepep_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    assert down_output.ndim == output.ndim == src2dst.ndim == 2
    assert output.ndim == topk_weights.ndim
    tokens, hidden = output.shape
    assert hidden_size == hidden and topk >= 0
    assert src2dst.shape == (tokens, topk)
    assert topk_weights.shape == (tokens, topk)
    assert down_output.shape[0] == tokens * topk
    assert down_output.shape[1] == hidden
    assert src2dst.dtype in (torch.int32, torch.int64)
    if not (tokens and topk and hidden):
        # Reference accumulates zeros; a skipped launch must not return
        # uninitialized memory (e.g. topk=0 with a non-empty output).
        return torch.zeros_like(output)
    out = torch.empty_like(output)
    _deepep_post_reorder[(min(tokens, 65535),)](
        down_output,
        out,
        src2dst,
        topk_weights,
        tokens,
        topk,
        hidden,
        down_output.stride(0),
        down_output.stride(1),
        out.stride(0),
        out.stride(1),
        src2dst.stride(0),
        src2dst.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        float(routed_scaling_factor),
        BLOCK=512,
    )
    return out


__all__ = ["deepep_post_reorder"]

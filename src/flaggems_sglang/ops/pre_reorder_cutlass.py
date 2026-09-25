# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 102 pre_reorder_cutlass: CUTLASS-MoE input permutation with a
# fused per-tensor dequant scale - every valid (token, slot) copies its
# scaled input row to the routed destination row of gateup_input
# (clone-first; untouched destinations keep base bytes; slots routed
# off-rank, topk_ids == num_local_experts, are skipped). One program
# per slot row; the optional 1/a1_scales factor is a scalar load so
# the None case compiles to a plain copy.

import torch
import triton
import triton.language as tl


@triton.jit
def _pre_reorder_kernel(
    src,
    scales,
    topk_ids,
    src2dst,
    out,
    num_local_experts,
    row_elems,
    topk,
    out_s0,
    is0,
    is1,
    ss0,
    ss1,
    HAS_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # branch-free (the T88-validated keep-factor pattern): a scalar if
    # on a loaded value faulted the launch on the proxy, so invalid
    # slots clamp their destination to row 0 and multiply by keep=0 -
    # the write becomes a harmless rewrite of an already-cloned row
    # the original async fault was the row-stride-as-flat-stride bug,
    # not this branch: off-rank slots must skip entirely because the
    # destination keeps its cloned base bytes
    slot = tl.program_id(0).to(tl.int64)
    t = slot // topk
    i = slot - t * topk
    eid = tl.load(topk_ids + t * is0 + i * is1)
    if eid != num_local_experts:
        dst = tl.load(src2dst + t * ss0 + i * ss1).to(tl.int64)
        inv = 1.0 / tl.load(scales).to(tl.float32) if HAS_SCALE else 1.0
        src_base = t * row_elems
        cols = tl.arange(0, BLOCK_C).to(tl.int64)
        for c0 in range(0, row_elems, BLOCK_C):
            cc = c0 + cols
            m = cc < row_elems
            v = tl.load(src + src_base + cc, mask=m, other=0).to(tl.float32)
            tl.store(
                out + dst * out_s0 + cc,
                (v * inv).to(out.dtype.element_ty),
                mask=m,
            )


def pre_reorder_cutlass(
    input,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
):
    out = gateup_input.clone()
    slots = num_tokens * topk
    if slots:
        row_elems = input.shape[-1]
        src = input if input.is_contiguous() else input.contiguous()
        _pre_reorder_kernel[(slots,)](
            src,
            a1_scales if a1_scales is not None else src,
            topk_ids,
            src2dst,
            out,
            num_local_experts,
            row_elems,
            topk,
            out.stride(0),
            topk_ids.stride(0),
            topk_ids.stride(1),
            src2dst.stride(0),
            src2dst.stride(1),
            HAS_SCALE=a1_scales is not None,
            BLOCK_C=1024,
            num_warps=4,
        )
    return out


__all__ = ["pre_reorder_cutlass"]

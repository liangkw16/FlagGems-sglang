# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 101 post_reorder_deepgemm: DeepGEMM MoE epilogue - the same
# gather-weighted-sum arithmetic as our 8/8-validated T88
# post_reorder_cutlass (346x team best) with the validity gate changed
# to topk_ids >= 0 (DeepGEMM pads with -1 and treats num_experts as
# the fused shared expert, so both stay valid here). src2dst entries
# of invalid slots are clamped to 0 before addressing (reference
# semantics).
#
# E1: hidden blocks move onto grid.y and BLOCK rises 1024 -> 2048 -
# the platform-validated T65-E10 sibling structure (deepep_post_reorder
# read +67.16% mean: tianshu x1.80 / muxi x1.49 / haiguang x1.96 /
# card_a x1.86 / card_b x1.54). gy = min(cdiv(hdim, BLOCK), 255) and
# gx = min(rows, 65535 // gy) keep total programs <= 65535; hidden is
# walked with a tl.range dual-axis stride. The arithmetic form is
# preserved from s0 byte-for-byte: TOPK static unroll, scalar slot
# reads, clamp + keep gate, fp32 accumulation, scale folded into the
# store, all strides as parameters. Kunlunxin runs the frozen s0
# vendor (wide BLOCK was a compile-stage SIGABRT there; see the
# deepep_post_reorder ledger).
#
# E3: invalid slots skip the whole gather. The scalar `if eid >= 0`
# branch carries the reference validity gate (topk_ids >= 0, the same
# predicate the branchless `keep` multiply encoded), so a padding slot
# no longer pays the full-block gather + the dst/weight scalar loads +
# the keep multiply - the SGLang main post_reorder_deepgemm form
# (ep_moe_kernels.py at 5f6dd44 wraps the weight+gather in
# `if dst_idx >= 0`). We deliberately keep the gate on topk_ids and the
# clamp: the reference includes down[clamp(dst)] * w for a VALID slot
# whose src2dst is -1, which upstream's dst-gate would drop. The token
# grid-stride row loop becomes tl.range(..., num_stages=3) software
# pipelining (upstream NUM_STAGES=3 on the same loop); the E1 geometry
# is untouched (grid.y hidden reorder, BLOCK=2048 / num_warps=8, fp32
# accumulation, scale folded into the store, parameterized strides).
# Divergence vs the literal torch reference is confined to non-finite
# data on INVALID slots (reference 0*NaN/Inf propagates NaN, the skip
# yields exactly 0); pinned in test_slot_skip_nonfinite_semantics and
# PREREGISTERED in the T101 ledger (E3 candidate record) as
# intentionally accepted - the platform harness data generator is not
# a documented contract, so "randn/finite" stays an unverified
# assumption guarded by the numeric-failure rollback gates (any
# numeric failure on a generic-rider chip rolls generic back to the
# e1 bytes). Loop-level num_stages has NO platform precedent on the
# generic riders' Triton forks: in-repo tl.range(num_stages=) exists
# only in _enflame vendor files (group_norm_silu E12 fired valid on
# enflame and read as a perf no-op there); a rider fork failing to
# compile or producing wrong numbers would invalidate the whole
# firing (T104 e2 precedent), so the NVIDIA proxy screens this as a
# disaster gate only and the platform is the arbiter.

import torch
import triton
import triton.language as tl

# e3 slot-skip marker consulted by tests/test_post_reorder_deepgemm.py
# to tell the skip semantics apart from the frozen s0 branchless bytes.
SLOT_SKIP_SEMANTICS = True


@triton.jit(do_not_specialize=["rows", "scale"])
def _post_reorder_deepgemm(
    down_output,
    src2dst,
    topk_ids,
    topk_weights,
    out,
    rows,
    scale,
    hdim,
    ds0,
    ss0,
    ss1,
    is0,
    is1,
    ws0,
    ws1,
    os0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in tl.range(
        tl.program_id(0), rows, tl.num_programs(0), num_stages=3
    ):
        base = row.to(tl.int64)
        obase = base * os0
        offs = tl.arange(0, BLOCK)
        for h0 in tl.range(
            tl.program_id(1) * BLOCK, hdim, tl.num_programs(1) * BLOCK
        ):
            ho = h0 + offs
            hm = ho < hdim
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for i in tl.static_range(0, TOPK):
                eid = tl.load(topk_ids + base * is0 + i * is1)
                if eid >= 0:
                    # e3 slot-skip: the reference validity gate as a
                    # scalar branch - invalid slots pay no gather and
                    # no dst/weight loads at all
                    dst = tl.load(src2dst + base * ss0 + i * ss1).to(
                        tl.int64
                    )
                    dst = tl.maximum(dst, 0)
                    w = tl.load(
                        topk_weights + base * ws0 + i * ws1
                    ).to(tl.float32)
                    v = tl.load(
                        down_output + dst * ds0 + ho, hm, other=0.0
                    ).to(tl.float32)
                    acc += v * w
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
    out = torch.empty(
        (rows, hdim), dtype=output.dtype, device=output.device
    )
    if rows and hdim:
        # E1: bounded hidden-grid launch (T65-E10 sibling form); the
        # grid product stays <= 65535 on every chip.
        grid_y = min(triton.cdiv(hdim, 2048), 255)
        grid_x = min(rows, max(1, 65535 // grid_y))
        _post_reorder_deepgemm[(grid_x, grid_y)](
            down_output,
            src2dst,
            topk_ids,
            topk_weights,
            out,
            rows,
            float(routed_scaling_factor),
            hdim,
            down_output.stride(0),
            src2dst.stride(0),
            src2dst.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            out.stride(0),
            TOPK=topk,
            BLOCK=2048,
            num_warps=8,
        )
    return out


__all__ = ["post_reorder_deepgemm"]

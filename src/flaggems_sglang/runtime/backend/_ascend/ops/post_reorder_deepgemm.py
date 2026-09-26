# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend vendor for Task 101 post_reorder_deepgemm (two-segment hidden loop).

The E1 generic bytes this file diverges from (source commit 0f9cf744,
ZIP e1-0f9cf74) carry two documented Ascend pathologies on EVERY hidden
block of the hot loop: `hm = ho < hdim` is an int32 vector compare that
Ascend Vector CMP lowers to scalar code (chip-rulesets.md:36), and
`tl.load(..., hm, other=0.0)` pre-fills the masked lanes and serializes
MTE2 (chip-rulesets.md:39). The platform gap pins the cost as structural
rather than a parameter axis: huawei reads 12.3874 (rank 4) against
OpeGoodn 22.567 (rank 1), -45.1%, the largest per-chip fixed-point
deficit on this task (climb-loop.json s2t1op101 gpu_results, re-read
this round); closing huawei r4 -> r1 alone is +0.75 ranking score under
the sum(1/rank) bookkeeping.

Structure: the T93 E3 two-segment template (add_constant.py:80-116,
fired there as submission 21510) migrated onto the E1 geometry. The
hidden tl.range loop keeps its grid.y striding, and each iteration
takes a uniform scalar branch `h0 + BLOCK <= hdim` (per-iteration
uniform; both arms are Triton compute, no PyTorch fallback, no
module-level mutable state): full hidden blocks load and store with NO
mask and NO other, so the hot path runs zero compares and zero prefill;
only the tail block of each row stride runs the masked load without
`other` (undef lanes never reach memory - the store carries the same
mask, the T92 e22 / upstream pointwise_dynamic form). Every other byte
of the arithmetic is preserved from E1: TOPK static unroll, scalar slot
reads, clamp + keep gate, fp32 accumulation, scale folded into the
store, all strides as parameters, grid.y hidden reorder with grid
product <= 65535, BLOCK=2048 / num_warps=8.

Family evidence (chain summarized in add_constant.py):
- T40 E16: hot path with zero per-tile compare, only the last
  sub-block masked -> huawei +130%; T39 e10: uniform scalar branch
  skipping whole blocks -> huawei +246%; T92 e22: dropping `other=`
  from the masked load kept numerics exact.
- T93 E3 (this template): valid submission 21510, huawei
  0.2472 -> 0.332 (+34%); T97 e3 (fused_sigmoid_mul, flat no-mask):
  valid submission 21516, huawei 1.674 (+28%, >=1.6 gate passed) -
  the no-mask direction is confirmed twice on the same Ascend silicon.
- T65 same-shape sibling: the persistent variant regressed huawei
  17.513 -> 14.1384 (-19.3%, deepep_post_reorder.md E13) - this
  candidate does NOT use persistence and keeps the E1 flat grid.

Counter-evidence disclosed at equal weight: T92 E23 (unpad) ran the
same concept (whole-block unmasked load/store plus a masked tail) on
the same huawei, passed 8/8 correct, yet read -28.8% vs e22 and was
rolled back (commit 47dc1382). Its difference surface: int64 addressing
on a (bs,tiles) 2D grid-stride loop with a per-base scalar guard. This
file keeps E1's int64 ROW addressing (base = row.to(tl.int64)) and
1D-per-axis grid, adding only the single hidden-axis scalar branch -
the family evidence cannot decompose which factor drove E23's
regression, so the huawei sign of this candidate is genuinely open.
The NVIDIA proxy cannot extrapolate the huawei launch-form response
either (session-mining-retrospective.md 4.1); the platform evaluator
is the arbiter. Runtime-branch compile constraints of the enflame class
do not apply here - this vendor is selected only on Ascend.

Preregistered gates (locked before screening/preflight): numeric
failure or huawei < 11.77 (-5%) -> roll back by deleting this vendor
file and restoring the E1 bytes; huawei >= 15 retains the vendor,
huawei >= 18 (clears GuanghuLab 17.5844 into the r2 band) confirms the
axis; platform average > 11.63934286 (current team best) swaps TB.
Firing requires the release matrix to proxy this file together with
the frozen kunlunxin vendor (--proxy-vendor ascend --proxy-vendor
kunlunxin). Huawei stays target-runtime-unverified until the platform
reading lands.

E3 (this round, fired out of the e2 [11.77, 15) middle band: huawei
12.7198, +2.68%, axis unconfirmed): both two-segment arms additionally
skip invalid slots whole - the scalar `if eid >= 0` branch carries the
reference validity gate (topk_ids >= 0) around the dst/weight loads
and the gather, the SGLang main post_reorder_deepgemm form
(ep_moe_kernels.py at 5f6dd44 wraps the weight+gather in
`if dst_idx >= 0`; we deliberately keep the gate on topk_ids plus the
clamp so a VALID slot with src2dst == -1 still contributes
down[clamp(-1)] * w, which the upstream dst-gate would drop). The token
grid-stride row loop becomes tl.range(..., num_stages=3) software
pipelining (upstream NUM_STAGES=3 on the same loop) - num_stages has
no Ascend-lowering precedent in this family, so compile risk rides the
NVIDIA proxy as a disaster gate only and the platform is the arbiter.
E1/E2 geometry preserved: grid.y hidden reorder, BLOCK=2048 /
num_warps=8, TOPK static unroll, fp32 accumulation, scale folded into
the store, parameterized strides. Divergence vs the literal torch
reference is confined to non-finite data on INVALID slots (reference
0*NaN/Inf propagates NaN, the skip yields exactly 0); pinned in
test_slot_skip_nonfinite_semantics and PREREGISTERED in the T101
ledger (E3 candidate record) as intentionally accepted - the platform
harness data generator is not a documented contract ("randn/finite"
stays an unverified assumption), and the same submission keeps the
frozen kunlunxin s0 branchless bytes, so the two backends
intentionally disagree on non-finite invalid-slot data within one
interface. Preregistered e3 gates (locked before screening): numeric
failure on huawei or huawei < 12.08 (e2 -5%) -> roll _ascend back to
the e2 bytes; numeric failure OR a -5% drop on ANY generic-rider chip
vs the e1 per-chip reading (tianshu 12.1168 / muxi 6.342 / haiguang
20.6092 / card_a 10.7918 / card_b 7.7752) -> roll generic back to the
e1 bytes - loop-level num_stages has no platform precedent on those
riders' Triton forks (in-repo tl.range(num_stages=) exists only in
_enflame vendor files; group_norm_silu E12 fired valid on enflame and
read as a perf no-op there), so a rider compile/numeric failure is a
live risk these gates convert into a byte rollback; platform average
> 11.63934286 swaps TB.
"""

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
            if h0 + BLOCK <= hdim:
                # hot path: the whole hidden block is in-bounds, so the
                # load/store run with no mask and no other -> no int32
                # Vector CMP, no MTE2 prefill on any full tile
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for i in tl.static_range(0, TOPK):
                    eid = tl.load(topk_ids + base * is0 + i * is1)
                    if eid >= 0:
                        # e3 slot-skip: the reference validity gate as
                        # a scalar branch - invalid slots pay no gather
                        # and no dst/weight loads at all
                        dst = tl.load(
                            src2dst + base * ss0 + i * ss1
                        ).to(tl.int64)
                        dst = tl.maximum(dst, 0)
                        w = tl.load(
                            topk_weights + base * ws0 + i * ws1
                        ).to(tl.float32)
                        v = tl.load(down_output + dst * ds0 + ho).to(
                            tl.float32
                        )
                        acc += v * w
                tl.store(
                    out + obase + ho,
                    (acc * scale).to(out.dtype.element_ty),
                )
            else:
                # tail block of this row stride: masked load without
                # `other` - undef lanes are never stored because the
                # store carries the same mask
                hm = ho < hdim
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for i in tl.static_range(0, TOPK):
                    eid = tl.load(topk_ids + base * is0 + i * is1)
                    if eid >= 0:
                        dst = tl.load(
                            src2dst + base * ss0 + i * ss1
                        ).to(tl.int64)
                        dst = tl.maximum(dst, 0)
                        w = tl.load(
                            topk_weights + base * ws0 + i * ws1
                        ).to(tl.float32)
                        v = tl.load(
                            down_output + dst * ds0 + ho, hm
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

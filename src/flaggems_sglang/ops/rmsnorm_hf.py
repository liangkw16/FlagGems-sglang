# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 104 rmsnorm_hf: HuggingFace LlamaRMSNorm semantics - the
# normalized activation is rounded back to the input dtype BEFORE the
# weight multiply (unlike the all-fp32 fused_rmsnorm). The kernel
# reproduces that order exactly: fp32 mean-of-squares reduction, fp32
# rsqrt scale, y rounded to the input dtype, then the weight multiply
# in fp32 with one final rounding at the store.
#
# E2 exact-unmask-whole-row: the S0 kernel was whole-row single-pass
# but ALWAYS masked - `cols < hidden` is an int32 vector compare that
# Ascend Vector CMP lowers to scalar code (chip-rulesets.md:36), and
# the two masked loads pre-fill `other`, serializing MTE2
# (chip-rulesets.md:39). For every pow2 hidden (64..8192, all present
# on the platform) BLOCK == hidden so the mask is all-true and that
# cost is pure overhead. EXACT is a compile-time constexpr branch
# (log_scaling_tau.py EVEN / T40 E16 pattern): pow2 shapes compile a
# fully naked kernel - zero bounds mask, zero `other` prefill, zero
# per-lane compare - while ragged hidden (333/1280/1536/3072/5120)
# keeps the mask but compares in fp32 (exact in-domain, < 2^24, the
# official Ascend CMP fix) and drops `other` from the weight load:
# undef w lanes only flow into the store, whose mask discards them,
# and w feeds no reduction. The x load keeps other=0.0 so masked
# lanes contribute 0 to the sum-of-squares.

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_hf_kernel(
    x,
    weight,
    out,
    x_s0,
    x_s1,
    w_s0,
    o_s0,
    hidden,
    eps,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if EXACT:
        # hidden == BLOCK (pow2): every lane of every row is in-bounds
        # by construction (strided x_s1/w_s0 walks stay inside the row
        # too), so all memory ops run fully naked - no bounds mask, no
        # `other` prefill, no per-lane compare.
        xv = tl.load(x + row * x_s0 + cols * x_s1).to(tl.float32)
        mean_square = tl.sum(xv * xv, axis=0) / hidden
        y = (xv * tl.rsqrt(mean_square + eps)).to(out.dtype.element_ty)
        w = tl.load(weight + cols * w_s0).to(tl.float32)
        tl.store(
            out + row * o_s0 + cols,
            (y.to(tl.float32) * w).to(out.dtype.element_ty),
        )
    else:
        # ragged hidden: compare in fp32 - exact for every in-domain
        # value (< 2^24, chip-rulesets.md:36) because Ascend Vector CMP
        # has no int32 form and degrades to scalar code. The weight
        # load drops `other`: undef lanes only flow into the store,
        # whose mask discards them, and w feeds no reduction. The x
        # load keeps other=0.0 to protect the sum-of-squares.
        m = cols.to(tl.float32) < hidden
        xv = tl.load(x + row * x_s0 + cols * x_s1, mask=m, other=0.0).to(
            tl.float32
        )
        mean_square = tl.sum(xv * xv, axis=0) / hidden
        y = (xv * tl.rsqrt(mean_square + eps)).to(out.dtype.element_ty)
        w = tl.load(weight + cols * w_s0, mask=m).to(tl.float32)
        tl.store(
            out + row * o_s0 + cols,
            (y.to(tl.float32) * w).to(out.dtype.element_ty),
            mask=m,
        )


def rmsnorm_hf(input, weight, eps):
    assert input.dim() == 2
    rows, hidden = input.shape
    # torch promotes weight * y per type rules: a wider weight dtype
    # widens the result beyond the input dtype
    out_dtype = torch.promote_types(weight.dtype, input.dtype)
    out = torch.empty(
        (rows, hidden), dtype=out_dtype, device=input.device
    )
    if rows and hidden:
        block = max(16, triton.next_power_of_2(hidden))
        _rmsnorm_hf_kernel[(rows,)](
            input,
            weight,
            out,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            out.stride(0),
            hidden,
            eps,
            BLOCK=block,
            # compile-time branch, not a runtime one: each pow2 hidden
            # compiles its own naked kernel; ragged shapes share the
            # masked variant
            EXACT=block == hidden,
            num_warps=8,
        )
    return out


__all__ = ["rmsnorm_hf"]

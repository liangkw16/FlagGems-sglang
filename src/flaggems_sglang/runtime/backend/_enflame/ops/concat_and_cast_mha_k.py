# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for concat_and_cast_mha_k, e3 structural round.
# Three GCU rules drive this form (FlagGems _enflame gcu300 codegen and
# FlagTree enflame backend, verified in the flagos-ai sources):
# 1. Runtime-valued strides keep FlagOfNotUseDMA from proving the
#    mutual-divisibility it requires, costing the 4x tile-shrink non-DMA
#    path - so every shape and stride is baked as constexpr (contiguous
#    layout asserted in the wrapper, matching the harness).
# 2. enable_i64=False: int64 offset arithmetic lowers to emulation, so
#    all addressing stays int32 (numel*itemsize is a few MB).
# 3. Official launch geometry: grid clamped to 12 CTAs with an in-kernel
#    grid-stride task loop and num_warps=2.
# The output interleaves a rope segment between consecutive heads, so the
# nope stores stay per-head rows (strides (DIM, 1), divisibility provable
# at compile time now that DIM is constexpr) - the flat-chunk shortcut
# across heads is wrong on the output side and was rejected in review.

import torch
import triton
import triton.language as tl

_MAX_CTAS = 12
_NUM_WARPS = 2
BH = 16  # heads per task group, passed to the kernel as a constexpr


@triton.jit(do_not_specialize=["tasks"])
def _concat_and_cast_mha_k(
    nope,
    rope,
    out,
    tasks,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    ND: tl.constexpr,
    RD: tl.constexpr,
    BNP: tl.constexpr,
    BR: tl.constexpr,
    DIM: tl.constexpr,
    BH: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        token = task // GROUPS
        g = task % GROUPS
        h = g * BH + tl.arange(0, BH)
        hm = h < HEADS
        n = tl.arange(0, BNP)
        r = tl.arange(0, BR)
        no = tl.load(
            nope + ((token * HEADS) + h[:, None]) * ND + n[None, :],
            hm[:, None] & (n[None, :] < ND),
            other=0,
        )
        ro = tl.load(rope + token * RD + r, r < RD, other=0)
        out_row = (token * HEADS + h[:, None]) * DIM
        tl.store(
            out + out_row + n[None, :],
            no.to(out.dtype.element_ty),
            hm[:, None] & (n[None, :] < ND),
        )
        tl.store(
            out + out_row + ND + r[None, :],
            ro.to(out.dtype.element_ty)[None, :],
            hm[:, None] & (r[None, :] < RD),
        )


def concat_and_cast_mha_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k_nope.dtype == k_rope.dtype
    assert k.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert k_nope.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not k_nope.is_contiguous():
        k_nope = k_nope.contiguous()
    if not k_rope.is_contiguous():
        k_rope = k_rope.contiguous()
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        assert out.numel() < 2**31  # int32 element offsets in the kernel
        groups = triton.cdiv(heads, BH)
        tasks = tokens * groups
        _concat_and_cast_mha_k[(min(tasks, _MAX_CTAS),)](
            k_nope,
            k_rope,
            out,
            tasks,
            HEADS=heads,
            GROUPS=groups,
            ND=nd,
            RD=rd,
            BNP=triton.next_power_of_2(max(1, nd)),
            BR=triton.next_power_of_2(max(1, rd)),
            DIM=dim,
            BH=BH,
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]

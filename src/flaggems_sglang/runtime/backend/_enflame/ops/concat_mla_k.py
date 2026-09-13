# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor, e6 revised per external review: the generic structure
# byte-for-byte (BH=16 shares one rope read across 16 heads - the e6
# draft's one-head-per-row form multiplied rope reads 16x), with the
# single change the recipe actually contributes on this chip: the task
# axis capped at the 24-SIP width (grid-stride body already present).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tasks", "heads", "groups", "nd", "rd"])
def _concat_mla_k(
    nope,
    rope,
    out,
    tasks,
    heads,
    groups,
    nd,
    rd,
    ns0,
    ns1,
    ns2,
    rs0,
    rs2,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        token = (task // groups).to(tl.int64)
        h = (task % groups) * BH + tl.arange(0, BH)
        n = tl.arange(0, BN).to(tl.int64)
        r = tl.arange(0, BR).to(tl.int64)
        no = tl.load(
            nope
            + token * ns0
            + h[:, None].to(tl.int64) * ns1
            + n[None, :] * ns2,
            (h[:, None] < heads) & (n[None, :] < nd),
            other=0,
        )
        ro = tl.load(rope + token * rs0 + r * rs2, r < rd, other=0)
        base = (token * heads + h.to(tl.int64)) * (nd + rd)
        tl.store(
            out + base[:, None] + n[None, :],
            no,
            (h[:, None] < heads) & (n[None, :] < nd),
        )
        tl.store(
            out + base[:, None] + nd + r[None, :],
            ro[None, :],
            (h[:, None] < heads) & (r[None, :] < rd),
        )


def concat_mla_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        # BH=16 quarters the program count per token (launch-bound shapes)
        # while re-reading each token's rope row once per group.
        groups = triton.cdiv(heads, 16)
        tasks = tokens * groups
        _concat_mla_k[(min(tasks, 24),)](
            k_nope,
            k_rope,
            out,
            tasks,
            heads,
            groups,
            nd,
            rd,
            *k_nope.stride(),
            k_rope.stride(0),
            k_rope.stride(2),
            BH=16,
            BN=triton.next_power_of_2(max(1, nd)),
            BR=triton.next_power_of_2(max(1, rd)),
        )
    return out


__all__ = ["concat_mla_k"]

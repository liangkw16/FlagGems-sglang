# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor, e10: the registered B2 de-mask plus today's cross-chip
# evidence - the same task's hygon vendor gained +25% from BH 16->4
# (submission 14686), so the tile here shrinks to 4 heads as well. When
# heads divides evenly by the 4-head tile
# and nd/rd exactly fill their power-of-two blocks, every load/store
# mask is compile-time true; this vendor drops them on that path
# (FlagTree's OffsetAnalysis treats per-lane bounds as burst-DMA
# blockers, and masked tails are the known miscompile family here).
# Any other shape keeps the masked e6 path, so the contract never
# narrows. Grid cap 24 with grid-stride and the shared rope read stay
# byte-identical to e6.

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
    NOMASK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    for task in range(tl.program_id(0), tasks, tl.num_programs(0)):
        token = (task // groups).to(tl.int64)
        h = (task % groups) * BH + tl.arange(0, BH)
        n = tl.arange(0, BN).to(tl.int64)
        r = tl.arange(0, BR).to(tl.int64)
        if NOMASK:
            no = tl.load(
                nope
                + token * ns0
                + h[:, None].to(tl.int64) * ns1
                + n[None, :] * ns2
            )
            ro = tl.load(rope + token * rs0 + r * rs2)
            base = (token * heads + h.to(tl.int64)) * (nd + rd)
            tl.store(out + base[:, None] + n[None, :], no)
            tl.store(out + base[:, None] + nd + r[None, :], ro[None, :])
        else:
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


# E11 probe: num_warps=4 launch pin (single variable; enflame 0.28 vs
# second-tier 1.23).
def concat_mla_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        groups = triton.cdiv(heads, 4)
        tasks = tokens * groups
        bn = triton.next_power_of_2(max(1, nd))
        br = triton.next_power_of_2(max(1, rd))
        nomask = heads % 4 == 0 and nd == bn and rd == br
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
            NOMASK=nomask,
            BH=4,
            BN=bn,
            BR=br,
            num_warps=4,
        )
    return out


__all__ = ["concat_mla_k"]

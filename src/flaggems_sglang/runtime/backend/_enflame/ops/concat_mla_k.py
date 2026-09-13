# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 8014d9d kernels/jit/csrc/elementwise/concat_mla.cuh.

# Enflame vendor, e6: the batch-5 recipe for this chip's streaming
# elementwise (BLOCK 4096 + 24-program grid cap + grid-stride, proved
# +92~113% on T73/T75/T71). concat_mla_k is exactly that shape: each
# output row is a 192-element copy assembled from two loads. The row
# axis caps and strides; BLOCK covers the padded row width.

import torch
import triton
import triton.language as tl

_MAX_PROGS = 24
_D_BLOCK = 256
_D_PAD: tl.constexpr = 256


@triton.jit
def _concat_mla_k_rows(
    nope,
    rope,
    out,
    rows,
    heads,
    nd,
    rd,
    ns0,
    ns1,
    ns2,
    rs0,
    rs2,
    os1,
    os2,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        token = row64 // heads
        head = row64 % heads
        base = row64 * os1
        src = token * ns0 + head * ns1
        d = tl.arange(0, BLOCK).to(tl.int64)
        nm = d < nd
        rm = (d >= nd) & (d < nd + rd)
        dn = tl.minimum(d, nd - 1)
        dr = tl.maximum(tl.minimum(d, nd + rd - 1) - nd, 0)
        no = tl.load(nope + src + dn * ns2, nm, other=0)
        ro = tl.load(rope + token * rs0 + dr * rs2, rm, other=0)
        tl.store(out + base + d * os2, no, nm)
        tl.store(out + base + d * os2, ro, rm)


def concat_mla_k(k, k_nope, k_rope):
    assert k.ndim == k_nope.ndim == k_rope.ndim == 3
    tokens, heads, dim = k.shape
    nd, rd = k_nope.shape[2], k_rope.shape[2]
    assert k_nope.shape[:2] == (tokens, heads)
    assert k_rope.shape[:2] == (tokens, 1) and nd + rd == dim
    assert k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if out.numel():
        rows = tokens * heads
        _concat_mla_k_rows[(min(rows, _MAX_PROGS),)](
            k_nope,
            k_rope,
            out,
            rows,
            heads,
            nd,
            rd,
            *k_nope.stride(),
            k_rope.stride(0),
            k_rope.stride(2),
            out.stride(1),
            out.stride(2),
            BLOCK=triton.next_power_of_2(max(1, dim)),
        )
    return out


__all__ = ["concat_mla_k"]

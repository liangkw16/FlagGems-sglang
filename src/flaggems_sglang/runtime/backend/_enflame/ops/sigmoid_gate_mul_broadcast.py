# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for sigmoid_gate_mul_broadcast: the T89 relu2 final
# streaming recipe ported to the broadcast gate - flat numel 1D
# grid-stride, min(cdiv(numel, 65536), 12) launch (12-CTA cap),
# BLOCK 65536 (relu2 ladder width top: 2.61@16384, 3.3@32768,
# 3.9@65536, 3.86@131072), num_stages 3, warps unpinned. Row forms are
# falsified both ways on GCU (0.77 with the 24-SIP cap, 0.63 with the
# full min(rows,2048) grid, field band 2.5-3.4). Gate is gathered per
# element via offs // HDIM (constexpr division) under the same tail
# mask; addressing is int32 with the domain asserted below.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel"])
def _sigmoid_gate_mul_broadcast(
    x, gate, out, numel,
    HDIM: tl.constexpr, BLOCK: tl.constexpr,
):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        # gate is (rows, 1) contiguous: stride(0) == 1, so the flat row
        # index offs // HDIM addresses it directly. The gather carries
        # the same m mask - masked lanes would read past gate's rows.
        g = tl.sigmoid(
            tl.load(gate + offs // HDIM, m, other=0.0).to(tl.float32)
        )
        v = tl.load(x + offs, m, other=0.0).to(tl.float32)
        tl.store(out + offs, (v * g).to(out.dtype.element_ty), m)


def sigmoid_gate_mul_broadcast(x, gate):
    assert x.ndim == 2
    rows, hdim = x.shape
    assert gate.shape == (rows, 1)
    assert gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    # flat streaming form: row-major contiguous inputs only (relu2
    # vendor precedent) and int32 addressing - the margin covers one
    # full tail block beyond numel so masked lanes cannot wrap.
    assert x.is_contiguous() and gate.is_contiguous()
    numel = rows * hdim
    assert numel < 2**31 - 65536
    out = torch.empty_like(x)
    if numel:
        # gcu300 grid cap (12 CTAs), relu2 final geometry; num_warps
        # stays unpinned - pinned narrow warps on super-wide blocks
        # were pathological even on the proxy (T89 e-final note).
        _sigmoid_gate_mul_broadcast[(min(triton.cdiv(numel, 65536), 12),)](
            x,
            gate,
            out,
            numel,
            HDIM=hdim,
            BLOCK=65536,
            num_stages=3,
        )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]

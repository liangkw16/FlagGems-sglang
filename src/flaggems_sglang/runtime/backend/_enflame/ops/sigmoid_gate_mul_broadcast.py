# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Enflame vendor for sigmoid_gate_mul_broadcast, e7 wholesale member
# replacement: the falsified forms are kept out by construction -
# e4/e5 row forms (0.77/0.63) walked rows with a runtime xs0 stride
# argument, and chip-rulesets.md:9 says runtime stride parameters fail
# the GCU DMA mutual-divisibility test (tile shrinks 4x, non-DMA);
# e6 flat streaming (0.033) gathered gate per element via
# offs // HDIM, a per-element division gather that relu2 (same chip,
# 3.94) never does. The e7 form tiles [RB, W] row blocks instead:
# W is the largest power-of-two factor of HDIM (capped at the 65536
# width-top), RB * W = 65536 keeps the tile on the relu2 width band,
# addressing is (base_row + arange(RB))[:, None] * HDIM +
# arange(W)[None, :] with HDIM constexpr (compile-time divisible row
# stride -> DMA path) plus a compile-time-counted column loop, gate is
# loaded once per row block as a contiguous [RB] vector, sigmoided and
# broadcast [RB, 1] - no runtime strides, no per-element division.
# Odd HDIM collapses W to 1: the [65536, 1] tile is correct but
# structurally slow (benchmark dims 2048/4096/5120/7168 all keep
# W >= 1024). Launch stays on the T89 recipe: grid-stride over row
# blocks with the gcu300 12-CTA cap, num_stages 3, warps unpinned.

# e7r carrier: same e7 execution bytes, re-rolled for the recovered kunlun water.
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _sigmoid_gate_mul_broadcast(
    x, gate, out, rows,
    HDIM: tl.constexpr, RB: tl.constexpr, W: tl.constexpr,
):
    for rb in range(
        tl.program_id(0), tl.cdiv(rows, RB), tl.num_programs(0)
    ):
        row_offs = rb * RB + tl.arange(0, RB)
        row_m = row_offs < rows
        # One contiguous [RB] gate vector per row block (no per-element
        # gather); sigmoid in fp32, broadcast down the column axis.
        g = tl.sigmoid(
            tl.load(gate + row_offs, row_m, other=0.0).to(tl.float32)
        )[:, None]
        m = row_m[:, None]
        # W divides HDIM by construction, so the column loop covers
        # HDIM exactly and no column mask ever exists; the row mask is
        # only live on the last (partial) row block. HDIM constexpr
        # keeps the [RB, W] tile strides (1, HDIM) compile-time
        # mutually divisible - the DMA requirement of chip-rulesets.
        for h0 in range(0, HDIM, W):
            offs = row_offs[:, None] * HDIM + (h0 + tl.arange(0, W))[
                None, :
            ]
            v = tl.load(x + offs, m, other=0.0).to(tl.float32)
            tl.store(out + offs, (v * g).to(out.dtype.element_ty), m)


def sigmoid_gate_mul_broadcast(x, gate):
    assert x.ndim == 2
    rows, hdim = x.shape
    assert gate.shape == (rows, 1)
    assert gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    # generic contract: gate contiguous, x may be row-gapped
    # (stride(1) == 1, stride(0) > hdim). The [RB, W] tile needs dense
    # rows, so row-gapped x takes a layout copy here (e6 P2
    # precedent) - the gating multiply still runs in the Triton kernel.
    assert gate.is_contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    numel = rows * hdim
    if numel:
        # W = largest power-of-two factor of hdim (hdim & -hdim),
        # capped at the 65536 relu2 width top; RB * W = 65536 exactly.
        # W | hdim always holds, so the kernel's column loop has no
        # tail and odd hdim degenerates to a correct-but-slow
        # [65536, 1] tile.
        w = min(hdim & -hdim, 65536)
        rb = 65536 // w
        # int32 addressing domain, e6-P1 style: the bound covers the
        # whole row-block induction, not just the accessed lanes -
        # visited row blocks stay < cdiv(rows, RB), so the largest
        # computed row index is cdiv(rows, RB) * RB - 1 <= rows + RB
        # - 1 and the largest computed flat address (masked tail lanes
        # are computed but never dereferenced) is
        # (rows + RB) * hdim - 1. A wrapped negative row_offs would
        # pass row_m and read/write at negative offsets, so keep every
        # computed address inside int32.
        assert (rows + rb) * hdim < 2**31
        # gcu300 grid cap (12 CTAs); num_warps stays unpinned - pinned
        # narrow warps on super-wide blocks were pathological even on
        # the proxy (T89 e-final note).
        _sigmoid_gate_mul_broadcast[(min(triton.cdiv(rows, rb), 12),)](
            x,
            gate,
            out,
            rows,
            HDIM=hdim,
            RB=rb,
            W=w,
            num_stages=3,
        )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]

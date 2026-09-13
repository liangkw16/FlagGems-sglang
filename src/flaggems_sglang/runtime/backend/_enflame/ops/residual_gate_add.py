# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 csrc/diffusion/residual_gate_add.cuh.

# Enflame vendor, e4 revised per external review: two kernels, one entry.
# - Same-shape gate: the flat-1D recipe form (BLOCK 4096, 24-program cap,
#   grid-stride, linear gate offsets).
# - Broadcast gate: keeps the e3 capped-2D form - the review correctly
#   noted the flat broadcast path would need `offs % d` (vector modulo
#   by a runtime scalar: no GCU precedent), and that the e1 "flat is
#   neutral" evidence only ever exercised the same-shape branch.

import torch
import triton
import triton.language as tl

_MAX_PROGS = 24
_BLOCK = 4096


@triton.jit
def _rga_flat(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    for pid in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        m = offs < n
        r = tl.load(r_ptr + offs, mask=m, other=0.0)
        u = tl.load(u_ptr + offs, mask=m, other=0.0)
        g = tl.load(g_ptr + offs, mask=m, other=0.0)
        # Multiply in the element dtype: an IEEE multiply is correctly
        # rounded, which equals the exact-fp32-then-round-to-dtype the
        # contract specifies. Going through fp32 explicitly lets the
        # compiler fold the round-trip away (caught by the precision
        # test: it skipped the intermediate rounding entirely).
        product = u * g
        out = (r.to(tl.float32) + product.to(tl.float32)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + offs, out, mask=m)


@triton.jit
def _rga_capped2d(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    rows,
    d,
    BLOCK: tl.constexpr,
):
    # The e3 form: row axis caps at 24 and strides, columns come from
    # program_id(1) (division-free); the broadcast gate reads by column.
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < d
        base = row.to(tl.int64) * d + cols
        r = tl.load(r_ptr + base, mask=mask, other=0.0)
        u = tl.load(u_ptr + base, mask=mask, other=0.0)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0)
        # Multiply in the element dtype: an IEEE multiply is correctly
        # rounded, which equals the exact-fp32-then-round-to-dtype the
        # contract specifies. Going through fp32 explicitly lets the
        # compiler fold the round-trip away (caught by the precision
        # test: it skipped the intermediate rounding entirely).
        product = u * g
        out = (r.to(tl.float32) + product.to(tl.float32)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + base, out, mask=mask)


def residual_gate_add(residual, update, gate):
    assert residual.dim() >= 2
    assert residual.shape == update.shape
    assert residual.is_contiguous() and update.is_contiguous()
    d = residual.shape[-1]
    n = residual.numel()
    if n == 0 or d == 0:
        return torch.empty_like(residual)
    rows = n // d
    if gate.shape == residual.shape:
        assert gate.is_contiguous()
        broadcast = False
    else:
        assert gate.shape == (1,) * (residual.dim() - 1) + (d,)
        broadcast = True
        gate = gate.reshape(-1)
        assert gate.is_contiguous()
    out = torch.empty_like(residual)
    if n:
        # fusion off everywhere: the double-rounding contract requires
        # the product to round to the dtype before the add; default FP
        # fusion folds that round-trip away (cancellation cases then
        # exceed the per-dtype tolerance - verified on the proxy).
        if broadcast:
            block = min(4096, triton.next_power_of_2(d))
            _rga_capped2d[(min(rows, _MAX_PROGS), triton.cdiv(d, block))](
                residual,
                update,
                gate,
                out,
                rows,
                d,
                BLOCK=block,
                enable_fp_fusion=False,
            )
        else:
            _rga_flat[(min(triton.cdiv(n, _BLOCK), _MAX_PROGS),)](
                residual,
                update,
                gate,
                out,
                n,
                BLOCK=_BLOCK,
                enable_fp_fusion=False,
            )
    return out


__all__ = ["residual_gate_add"]

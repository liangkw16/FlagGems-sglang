# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Triton port of SGLang fd32226 csrc/diffusion/residual_gate_add.cuh.

# Kunlunxin vendor, e5: both paths rebuilt without runtime loops - the
# mechanism that took T75's kunlun reading from 0.26 to 0.90 on this
# same diffusion family. The broadcast path keeps the 2D (row, col
# block) grid; the same-shape path launches exactly cdiv(n, BLOCK)
# programs with straight-line bodies (no grid-stride). BLOCK stays
# 1024 and the double-rounding contract keeps enable_fp_fusion=False:
# the product rounds to the tensor dtype before the residual add,
# matching the eager chain (fp fusion folds that round-trip away and
# blows the tolerance - caught by the cancellation cases).

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _residual_gate_add_bc(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    base = row * d + cols
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    u = tl.load(u_ptr + base, mask=mask, other=0.0)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0)
    product = (u.to(tl.float32) * g.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    out = (r.to(tl.float32) + product.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + base, out, mask=mask)


@triton.jit
def _residual_gate_add_flat_kx(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = offs < n
    r = tl.load(r_ptr + offs, mask=m, other=0.0)
    u = tl.load(u_ptr + offs, mask=m, other=0.0)
    g = tl.load(g_ptr + offs, mask=m, other=0.0)
    # Multiply in the element dtype: an IEEE multiply is correctly
    # rounded, which equals the exact-fp32-then-round-to-dtype the
    # contract specifies; an explicit fp32 detour lets fusion fold the
    # round-trip away (cancellation cases exceed the tolerance).
    product = u * g
    out = (r.to(tl.float32) + product.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + offs, out, mask=m)


def residual_gate_add(residual, update, gate):
    assert residual.dim() >= 2
    assert residual.shape == update.shape
    assert residual.is_contiguous() and update.is_contiguous()
    d = residual.shape[-1]
    if residual.numel() == 0 or d == 0:
        return torch.empty_like(residual)
    rows = residual.numel() // d
    if gate.shape == residual.shape:
        assert gate.is_contiguous()
        broadcast = False
    else:
        assert gate.shape == (1,) * (residual.dim() - 1) + (d,)
        broadcast = True
        gate = gate.reshape(-1)
        assert gate.is_contiguous()
    out = torch.empty_like(residual)
    if broadcast:
        _residual_gate_add_bc[(rows, triton.cdiv(d, _BLOCK))](
            residual,
            update,
            gate,
            out,
            d,
            BLOCK=_BLOCK,
            enable_fp_fusion=False,
        )
    else:
        n = residual.numel()
        _residual_gate_add_flat_kx[(triton.cdiv(n, _BLOCK),)](
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

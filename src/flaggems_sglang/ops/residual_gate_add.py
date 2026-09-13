# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Triton port of SGLang fd32226 csrc/diffusion/residual_gate_add.cuh.

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _residual_gate_add(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    d,
    BROADCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    base = row * d + cols
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    u = tl.load(u_ptr + base, mask=mask, other=0.0)
    if BROADCAST:
        g = tl.load(g_ptr + cols, mask=mask, other=0.0)
    else:
        g = tl.load(g_ptr + base, mask=mask, other=0.0)
    # The double rounding is the contract: round the product to the
    # tensor dtype before the residual add, matching the eager chain.
    product = (u.to(tl.float32) * g.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    out = (r.to(tl.float32) + product.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + base, out, mask=mask)


@triton.jit
def _residual_gate_add_flat(
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
    # The product must round to the tensor dtype before the residual
    # add (the task's double-rounding contract). Triton's default FP
    # fusion folds the register-level round-trip away - verified with
    # cancellation cases that then EXCEED the per-dtype tolerance -
    # so every launch disables the fusion.
    if broadcast:
        _residual_gate_add[(rows, triton.cdiv(d, _BLOCK))](
            residual,
            update,
            gate,
            out,
            d,
            BROADCAST=broadcast,
            BLOCK=_BLOCK,
            enable_fp_fusion=False,
        )
    else:
        # Same-shape gate: flat 1D avoids the (rows, d/BLOCK) grid
        # wasting programs on narrow rows.
        n = residual.numel()
        _residual_gate_add_flat[(min(triton.cdiv(n, _BLOCK), 65535),)](
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

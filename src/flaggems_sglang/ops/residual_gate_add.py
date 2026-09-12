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


def residual_gate_add(residual, update, gate):
    assert residual.dim() >= 2
    assert residual.shape == update.shape
    assert residual.is_contiguous() and update.is_contiguous()
    d = residual.shape[-1]
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
    if rows and d:
        _residual_gate_add[(rows, triton.cdiv(d, _BLOCK))](
            residual,
            update,
            gate,
            out,
            d,
            BROADCAST=broadcast,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["residual_gate_add"]

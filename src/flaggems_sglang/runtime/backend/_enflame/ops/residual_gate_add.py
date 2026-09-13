# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 csrc/diffusion/residual_gate_add.cuh.

# Enflame vendor: the skill-documented oversubscription fix (grid capped
# at the 24-SIP width, T19-E5/T51-E5 platform precedent +38% median).
# The row axis caps at 24 programs and strides; the column axis keeps
# its natural block count. Division-free addressing (the row index is
# the loop variable, columns come from program_id(1)). BLOCK adapts to
# the row width so narrow rows keep lane utilization. Semantics are
# identical to the generic (double rounding preserved).

import torch
import triton
import triton.language as tl

_MAX_ROWS = 24


@triton.jit
def _residual_gate_add_capped(
    r_ptr,
    u_ptr,
    g_ptr,
    out_ptr,
    rows,
    d,
    BROADCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < d
        base = row.to(tl.int64) * d + cols
        r = tl.load(r_ptr + base, mask=mask, other=0.0)
        u = tl.load(u_ptr + base, mask=mask, other=0.0)
        if BROADCAST:
            g = tl.load(g_ptr + cols, mask=mask, other=0.0)
        else:
            g = tl.load(g_ptr + base, mask=mask, other=0.0)
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
        block = min(1024, triton.next_power_of_2(d))
        _residual_gate_add_capped[
            (min(rows, _MAX_ROWS), triton.cdiv(d, block))
        ](
            residual,
            update,
            gate,
            out,
            rows,
            d,
            BROADCAST=broadcast,
            BLOCK=block,
        )
    return out


__all__ = ["residual_gate_add"]

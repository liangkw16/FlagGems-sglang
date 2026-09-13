# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang fd32226 csrc/diffusion/residual_gate_add.cuh.

# Enflame vendor, e4: the e3 recipe (BLOCK 4096 + 24-program cap) is
# already proven on this chip's streaming elementwise. E4 pushes further
# with the flat-1D form the e1 verdict showed is neutral on every chip:
# a single flattened grid-stride over rows*cols eliminates the 2D grid's
# narrow-row program waste AND the row-loop, keeping one store per
# element. Gate indexing switches on BROADCAST at compile time - the
# broadcast gate re-reads by modulo-free flat offsets only when the
# shapes force it (same-shape gate reads linearly like r/u).

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
    d,
    BROADCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for pid in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        m = offs < n
        r = tl.load(r_ptr + offs, mask=m, other=0.0)
        u = tl.load(u_ptr + offs, mask=m, other=0.0)
        if BROADCAST:
            g = tl.load(g_ptr + offs % d, mask=m, other=0.0)
        else:
            g = tl.load(g_ptr + offs, mask=m, other=0.0)
        product = (u.to(tl.float32) * g.to(tl.float32)).to(
            out_ptr.dtype.element_ty
        )
        out = (r.to(tl.float32) + product.to(tl.float32)).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + offs, out, mask=m)


def residual_gate_add(residual, update, gate):
    assert residual.dim() >= 2
    assert residual.shape == update.shape
    assert residual.is_contiguous() and update.is_contiguous()
    d = residual.shape[-1]
    n = residual.numel()
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
        _rga_flat[(min(triton.cdiv(n, _BLOCK), _MAX_PROGS),)](
            residual,
            update,
            gate,
            out,
            n,
            d,
            BROADCAST=broadcast,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["residual_gate_add"]

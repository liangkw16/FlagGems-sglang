# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Hygon vendor reverted to generic bytes (warps16 read -10% on
# submission 18315; the T86-positive recipe does not transfer here).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _sigmoid_gate_mul_broadcast(
    x, gate, out, rows, xs0, gs0, os0,
    HDIM: tl.constexpr, BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64)
        g = tl.sigmoid(tl.load(gate + row * gs0).to(tl.float32))
        for h0 in tl.static_range(0, HDIM, BLOCK):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < HDIM
            v = tl.load(x + base * xs0 + offs, m, other=0.0).to(
                tl.float32
            )
            tl.store(
                out + base * os0 + offs,
                (v * g).to(out.dtype.element_ty),
                m,
            )


def sigmoid_gate_mul_broadcast(x, gate):
    assert x.ndim == 2
    rows, hdim = x.shape
    assert gate.shape == (rows, 1)
    assert gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.stride(1) == 1 and gate.is_contiguous()
    out = torch.empty_like(x)
    if rows and hdim:
        _sigmoid_gate_mul_broadcast[(min(rows, 2048),)](
            x,
            gate,
            out,
            rows,
            x.stride(0),
            gate.stride(0),
            out.stride(0),
            HDIM=hdim,
            BLOCK=min(1024, triton.next_power_of_2(max(1, hdim))),
        )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]

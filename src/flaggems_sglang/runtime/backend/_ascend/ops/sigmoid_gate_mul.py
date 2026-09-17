# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor: persistent launch kept close to the physical Vector
# Core count (Ascend vector_operator.md: GPU-style huge grids pay
# repeated dispatch overhead on NPUs; each program strides over tiles
# in an inner loop - decode_attention template). The kernel body and
# BLOCK stay identical to the generic bytes; the warps pin from the
# e9 probe is dropped (measured flat there, unpinned on T64's
# persistent carrier).

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

_DEFAULT_VECTOR_CORES = 40


def _get_num_vector_cores(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties.get("num_vectorcore", _DEFAULT_VECTOR_CORES))


_BLOCK = 4096


@triton.jit
def _sigmoid_gate_mul(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    for pid in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * (1.0 / (1.0 + tl.exp(-g)))
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x, gate):
    assert x.shape == gate.shape
    assert x.is_contiguous() and gate.is_contiguous()
    n = x.numel()
    out = torch.empty_like(x)
    if n:
        workers = _get_num_vector_cores(x.device.index)
        _sigmoid_gate_mul[(min(triton.cdiv(n, _BLOCK), workers),)](
            x, gate, out, n, BLOCK=_BLOCK
        )
    return out


__all__ = ["sigmoid_gate_mul"]

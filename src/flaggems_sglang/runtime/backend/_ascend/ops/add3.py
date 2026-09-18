# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for add3: persistent launch kept close to the physical
# Vector Core count (GPU-style huge grids pay repeated dispatch
# overhead on NPUs); kernel bytes are the generic form and the
# grid-stride loop already lives in the body.

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

_DEFAULT_VECTOR_CORES = 40


def _get_num_vector_cores(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties.get("num_vectorcore", _DEFAULT_VECTOR_CORES))


def _worker_count(tensor):
    index = getattr(getattr(tensor, "device", None), "index", None)
    if index is None:
        return _DEFAULT_VECTOR_CORES
    return _get_num_vector_cores(index)


@triton.jit
def _add3(a, b, c, out, numel, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        av = tl.load(a + offs, m, other=0).to(tl.float32)
        bv = tl.load(b + offs, m, other=0).to(tl.float32)
        cv = tl.load(c + offs, m, other=0).to(tl.float32)
        mid = (av + bv).to(out.dtype.element_ty).to(tl.float32)
        tl.store(out + offs, (mid + cv).to(out.dtype.element_ty), m)


def add3(a, b, c):
    assert a.dtype == b.dtype == c.dtype == torch.bfloat16
    assert a.shape == b.shape == c.shape
    for tensor in (a, b, c):
        assert tensor.is_contiguous()
    numel = a.numel()
    assert numel % 16 == 0
    out = torch.empty_like(a)
    if numel:
        _add3[(min(triton.cdiv(numel, 1024), _worker_count(a)),)](
            a, b, c, out, numel, BLOCK=1024
        )
    return out


__all__ = ["add3"]

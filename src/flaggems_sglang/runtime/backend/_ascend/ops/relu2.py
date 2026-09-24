# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for relu2 (e9 form, e9r carrier): persistent launch capped at the
# Vector Core count (copy-family recipe) on the >=4096 ladder rung
# (BLOCK=4096, num_warps=16) with no masked-load other prefill - the
# other=0.0 prefill serializes MTE2 on Ascend (chip-rulesets) and the
# masked lanes never reach the equally-masked store.

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




@triton.jit(do_not_specialize=["numel"])
def _relu2(x, out, numel, BLOCK: tl.constexpr):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        v = tl.load(x + offs, m).to(tl.float32)
        # NaN-preserving relu: tl.maximum(nan, 0) returns 0 on this
        # backend; the where form keeps NaN (nan < 0 is False).
        r = tl.where(v < 0.0, 0.0, v)
        tl.store(out + offs, (r * r).to(out.dtype.element_ty), m)


def relu2(input):
    assert input.ndim == 2
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert input.is_contiguous()
    out = torch.empty_like(input)
    numel = input.numel()
    if numel:
        _relu2[(min(triton.cdiv(numel, 4096), _worker_count(input)),)](
            input, out, numel, BLOCK=4096, num_warps=16
        )
    return out


__all__ = ["relu2"]

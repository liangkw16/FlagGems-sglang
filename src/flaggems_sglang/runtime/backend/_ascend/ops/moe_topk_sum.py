# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for moe_topk_sum: persistent launch capped at the
# physical Vector Core count (the copy-family recipe); kernel bytes are
# the generic form and the row grid-stride already lives in the body.

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


@triton.jit(do_not_specialize=["rows", "hdim"])
def _moe_topk_sum(x, out, rows, hdim, TOPK: tl.constexpr,
                  BLOCK: tl.constexpr):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        base = row.to(tl.int64) * (TOPK * hdim)
        for h0 in tl.range(
            tl.program_id(1) * BLOCK, hdim, tl.num_programs(1) * BLOCK
        ):
            offs = h0 + tl.arange(0, BLOCK)
            m = offs < hdim
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for t in tl.static_range(0, TOPK):
                acc += tl.load(
                    x + base + t * hdim + offs,
                    m,
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                out + row.to(tl.int64) * hdim + offs,
                acc.to(out.dtype.element_ty),
                m,
            )




def moe_topk_sum(x, out):
    assert x.ndim == 3 and out.ndim == 2
    rows, topk, hdim = x.shape
    assert out.shape == (rows, hdim)
    assert x.dtype == out.dtype == torch.bfloat16
    assert x.is_contiguous() and out.is_contiguous()
    if rows and hdim:
        splits = min(max(1, triton.cdiv(hdim, 1024)), 255)
        _moe_topk_sum[(min(rows, _worker_count(out)), splits)](
            x,
            out,
            rows,
            hdim,
            TOPK=topk,
            BLOCK=1024,
        )
    return out


__all__ = ["moe_topk_sum"]

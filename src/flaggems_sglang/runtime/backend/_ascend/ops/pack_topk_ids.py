# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for pack_topk_ids: persistent launch capped at the
# physical Vector Core count (the copy-family recipe) with the bf16
# truncation computed as pure integer round-to-nearest-even bit math -
# (u32 + 0x7FFF + lsb(u32 >> 16)) >> 16 - instead of the
# f32 -> bf16 -> f32 convert chain, so the bit pattern never touches a
# float convert (exact for finite values, +/-Inf and quiet NaN; no
# masked-load other prefill, MTE2 serialization avoided per
# chip-rulesets).

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
def _pack_topk_ids_ascend(
    topk_ids, topk_weights, out, numel, BLOCK: tl.constexpr,
):
    for base in range(
        tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK
    ):
        offs = base + tl.arange(0, BLOCK)
        m = offs < numel
        ids = tl.load(topk_ids + offs, m)
        w = tl.load(topk_weights + offs, m)
        u = w.to(tl.int32, bitcast=True)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
        tl.store(out + offs, (ids << 16) | bits, m)


def pack_topk_ids(topk_ids, topk_weights):
    assert topk_ids.shape == topk_weights.shape
    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    out = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    if numel:
        workers = _worker_count(topk_ids)
        _pack_topk_ids_ascend[(min(triton.cdiv(numel, 4096), workers),)](
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK=4096,
            num_warps=16,
        )
    return out


__all__ = ["pack_topk_ids"]

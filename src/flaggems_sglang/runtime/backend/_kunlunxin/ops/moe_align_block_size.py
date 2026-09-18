# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Kunlun vendor for moe_align_block_size: XPU rejects both the atomic
# cursor ('atomic_rmw destroyed but still has uses') and tl.cumsum
# ('tt.scan explicitly illegal'), so a single program places every
# element with pure scalar loops - no vectors, no atomics, no scans.
# Complexity matches the reference (O(E * numel) on-device vs Python).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel", "num_routed", "block_size",
                               "buf_numel"])
def _moe_align(
    flat,
    sorted_ids,
    expert_ids,
    num_post,
    numel,
    num_routed,
    block_size,
    buf_numel,
    BLOCK: tl.constexpr,
):
    for base in range(0, buf_numel, BLOCK):
        o = base + tl.arange(0, BLOCK)
        v = tl.full((BLOCK,), numel, dtype=tl.int32)
        tl.store(sorted_ids + o, v, o < buf_numel)
    offset = 0
    for e in range(0, num_routed):
        count = 0
        for i in range(0, numel):
            v = tl.load(flat + i)
            if v == e:
                tl.store(sorted_ids + offset + count, i)
                count += 1
        aligned = ((count + block_size - 1) // block_size) * block_size
        beg = offset // block_size
        for b in range(0, aligned // block_size):
            tl.store(expert_ids + beg + b, e)
        offset += aligned
    tl.store(num_post, offset)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    flat = topk_ids.reshape(-1)
    numel = flat.numel()
    num_routed = num_experts - 1
    sorted_ids = torch.empty_like(sorted_token_ids)
    eids = expert_ids.clone()
    npost = torch.empty_like(num_tokens_post_pad)
    _moe_align[(1,)](
        flat,
        sorted_ids,
        eids,
        npost,
        numel,
        num_routed,
        block_size,
        sorted_ids.numel(),
        BLOCK=1024,
    )
    return sorted_ids, eids, npost


__all__ = ["moe_align_block_size"]

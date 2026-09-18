# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# kunlunxin vendor for moe_align_block_size: the atomic-cursor scatter
# breaks this backend (uni_sram: atomic_rmw destroyed-but-used plus resource overflow), so a single deterministic program
# does everything - sentinel fill, per-expert scan with an exclusive
# tl.cumsum for in-chunk placement (no atomics), expert blocks and the
# post-pad count. Complexity matches the reference (O(E * numel)).

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["numel", "num_routed", "block_size", "buf_numel"])
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
    fill = tl.full((BLOCK,), numel, dtype=tl.int32)
    for base in range(0, buf_numel, BLOCK):
        o = base + tl.arange(0, BLOCK)
        tl.store(sorted_ids + o, fill, o < buf_numel)
    offset = 0
    for e in range(0, num_routed):
        carry = tl.zeros((), dtype=tl.int32)
        for base in range(0, numel, BLOCK):
            offs = base + tl.arange(0, BLOCK)
            m = offs < numel
            v = tl.load(flat + offs, m, other=-1)
            hit = (v == e) & m
            h = hit.to(tl.int32)
            pref = tl.cumsum(h, axis=0) - h
            tl.store(sorted_ids + offset + carry + pref, offs, hit)
            carry += tl.sum(h, axis=0)
        n = carry
        aligned = ((n + block_size - 1) // block_size) * block_size
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

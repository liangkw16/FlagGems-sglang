# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Short-rank GEMV: device routing plus FP32 elementwise products. Keeping
# lane partials across K blocks moves the cross-thread sum outside the loop
# (vLLM PR 52880, commit 3d45361674f874eccf51f04999e17e5f0b28c3b4).
import torch
import triton
import triton.language as tl


@triton.jit
def _expand_route(
    seg,
    route,
    S: tl.constexpr,
    BS: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), BS + 1, tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        bound = tl.load(seg + tl.minimum(mid, BS) * SEG_STRIDE)
        right = (lo < hi) & (bound <= pos)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    owner = lo - 1
    start = tl.load(seg)
    end = tl.load(seg + BS * SEG_STRIDE)
    active = (pos >= start) & (pos < end) & (owner >= 0) & (owner < BS)
    tl.store(route + pos, tl.where(active, owner, -1), mask=pos < S)


@triton.jit
def _expand_vector(
    x,
    w,
    base,
    out,
    route,
    perm,
    wid,
    ranks,
    scales,
    slices,
    task_start,
    S: tl.constexpr,
    NS: tl.constexpr,
    NTILES: tl.constexpr,
    NUM_LORA: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    PS: tl.constexpr,
    IS: tl.constexpr,
    RS: tl.constexpr,
    SS: tl.constexpr,
    SLS: tl.constexpr,
    RANK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = task_start + tl.program_id(0)
    tile = task % NTILES
    sid = (task // NTILES) % NS
    pos = task // (NTILES * NS)
    owner = tl.load(route + pos)
    idx = tl.load(wid + tl.maximum(owner, 0) * IS)
    safe_idx = tl.minimum(tl.maximum(idx, 0), NUM_LORA - 1)
    live = (owner >= 0) & (tl.load(ranks + safe_idx * RS) != 0)
    row = tl.load(perm + pos * PS).to(tl.int64)
    begin = tl.load(slices + sid * SLS)
    end = tl.load(slices + (sid + 1) * SLS)
    col = begin + tile * BN + tl.arange(0, BN)
    kval = tl.arange(0, BK)
    partial = tl.zeros((BN, BK), tl.float32)
    for kbase in range(0, RANK, BK):
        k = kbase + kval
        xv = tl.load(
            x + row * XS0 + (sid * RANK + k) * XS1,
            mask=(k < RANK) & live,
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(
            w
            + safe_idx.to(tl.int64) * WS0
            + col[:, None].to(tl.int64) * WS1
            + k[None, :] * WS2,
            mask=(col[:, None] < end) & (k[None, :] < RANK) & live,
            other=0.0,
        ).to(tl.float32)
        partial += wv * xv[None, :]
    value = tl.sum(partial, axis=1)
    scale = tl.load(scales + safe_idx * SS).to(tl.float32)
    old = tl.load(
        base + row * BS0 + col * BS1, mask=(col < end) & live, other=0.0
    ).to(tl.float32)
    tl.store(
        out + row * OS0 + col * OS1,
        old + value * scale,
        mask=(col < end) & live,
    )


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    output = base_output.clone()
    ns = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    if x.shape[1] != ns * rank:
        raise ValueError("x width must equal n_slices * rank")
    total = batch_info.permutation.numel()
    if output.numel() == 0 or total == 0 or ns <= 0 or batch_info.bs == 0:
        return output
    route = torch.empty((total,), dtype=torch.int32, device=x.device)
    _expand_route[(triton.cdiv(total, 256),)](
        batch_info.seg_indptr,
        route,
        total,
        batch_info.bs,
        batch_info.seg_indptr.stride(0),
        (batch_info.bs + 1).bit_length(),
        BLOCK=256,
        num_warps=4,
    )
    bn = 16
    bk = min(triton.next_power_of_2(max(rank, 1)), 128)
    tiles = triton.cdiv(int(max_slice_size), bn)
    tasks = total * ns * tiles
    # Shape-only chunks respect the backend grid cap without a runtime
    # persistent loop. Metadata is recomputed from current tensors each call.
    for start in range(0, tasks, 65535):
        _expand_vector[(min(tasks - start, 65535),)](
            x,
            weights,
            base_output,
            output,
            route,
            batch_info.permutation,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            slice_offsets,
            start,
            total,
            ns,
            tiles,
            weights.shape[0],
            *x.stride(),
            *weights.stride(),
            *base_output.stride(),
            *output.stride(),
            batch_info.permutation.stride(0),
            batch_info.weight_indices.stride(0),
            batch_info.lora_ranks.stride(0),
            batch_info.scalings.stride(0),
            slice_offsets.stride(0),
            rank,
            bn,
            bk,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output


__all__ = ["chunked_sgmv_expand"]

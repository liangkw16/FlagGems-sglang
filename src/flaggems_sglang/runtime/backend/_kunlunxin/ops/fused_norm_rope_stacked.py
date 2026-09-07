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

# Kunlunxin vendor: three-kernel split mirroring platform-proven forms
# from this backend. (A) RMSNorm row kernel in the T51-proven shape
# (one program per (t, l, h), 1D [BLOCK_D], tl.rsqrt) writing an fp32
# intermediate; (B) pair RoPE in the T49-proven shape reading the fp32
# intermediate and casting once to the output dtype; (C) a plain
# contiguous row copy for V. Every mask is a single 1D predicate: the
# fused e1/e2 forms left elements wildly wrong here (compound
# `d_mask & (offs_d >= rotary_dim)` masks and per-kernel V tails are
# the differences), and the E16/T45 lesson bans compound-predicate
# load/store on this backend.

import torch
import triton
import triton.language as tl


@triton.jit
def _k_norm_fp32_kernel(
    kv_ptr,
    w_ptr,
    eps_ptr,
    mid_ptr,
    T,
    num_layers,
    num_heads,
    head_dim,
    kv_stride_t,
    kv_stride_l,
    w_stride_l,
    eps_stride,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % num_heads
    tl_idx = row // num_heads
    l = tl_idx % num_layers
    t = tl_idx // num_layers

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    kv_base = t * kv_stride_t + l * kv_stride_l + h * head_dim
    x = tl.load(kv_ptr + kv_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    eps = tl.load(eps_ptr + l * eps_stride).to(tl.float32)
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + l * w_stride_l + offs_d, mask=d_mask, other=1.0).to(tl.float32)
    mid = x * inv_rms * w
    mid_base = ((l * T + t) * num_heads + h) * head_dim
    tl.store(mid_ptr + mid_base + offs_d, mid, mask=d_mask)


@triton.jit
def _k_rope_cast_kernel(
    mid_ptr,
    cos_sin_ptr,
    pos_ptr,
    k_out_ptr,
    T,
    num_layers,
    num_heads,
    head_dim,
    rotary_dim,
    half_rd,
    tail_len,
    cs_stride,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % num_heads
    tl_idx = row // num_heads
    l = tl_idx % num_layers
    t = tl_idx // num_layers
    base = ((l * T + t) * num_heads + h) * head_dim
    out_ty = k_out_ptr.dtype.element_ty

    if HAS_ROPE:
        r = tl.arange(0, BLOCK_R)
        r_mask = r < half_rd
        pos = tl.load(pos_ptr + t)
        cs = cos_sin_ptr + pos.to(tl.int64) * cs_stride
        cos = tl.load(cs + r, mask=r_mask, other=0.0).to(tl.float32)
        sin = tl.load(cs + r + half_rd, mask=r_mask, other=0.0).to(tl.float32)
        k1 = tl.load(mid_ptr + base + r, mask=r_mask, other=0.0)
        k2 = tl.load(mid_ptr + base + r + half_rd, mask=r_mask, other=0.0)
        rot1 = k1 * cos - k2 * sin
        rot2 = k2 * cos + k1 * sin
        tl.store(k_out_ptr + base + r, rot1.to(out_ty), mask=r_mask)
        tl.store(k_out_ptr + base + r + half_rd, rot2.to(out_ty), mask=r_mask)

    if HAS_TAIL:
        td = tl.arange(0, BLOCK_TAIL)
        t_mask = td < tail_len
        tail = tl.load(mid_ptr + base + rotary_dim + td, mask=t_mask, other=0.0)
        tl.store(k_out_ptr + base + rotary_dim + td, tail.to(out_ty), mask=t_mask)


@triton.jit
def _v_copy_kernel(
    kv_ptr,
    v_out_ptr,
    T,
    num_layers,
    kv_size,
    kv_stride_t,
    kv_stride_l,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    l = row % num_layers
    t = row // num_layers
    offs = tl.arange(0, BLOCK_V)
    mask = offs < kv_size
    v = tl.load(
        kv_ptr + t * kv_stride_t + l * kv_stride_l + kv_size + offs,
        mask=mask,
        other=0,
    )
    tl.store(v_out_ptr + (l * T + t) * kv_size + offs, v, mask=mask)


def fused_norm_rope_stacked(
    kv, k_norm_weight, eps, cos_sin_cache, positions, num_kv_heads, head_dim, rotary_dim
):
    T, L, width = kv.shape
    H, D = num_kv_heads, head_dim
    kv_size = H * D
    k_out = torch.empty((L, T, H, D), dtype=kv.dtype, device=kv.device)
    v_out = torch.empty((L, T, H, D), dtype=kv.dtype, device=kv.device)
    if T * L == 0 or kv_size == 0:
        return k_out, v_out

    kv = kv.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    positions = positions.contiguous()
    w = k_norm_weight.contiguous().reshape(-1, D)
    eps_t = (
        eps.contiguous().reshape(-1)
        if isinstance(eps, torch.Tensor)
        else torch.full((L,), float(eps), dtype=torch.float32, device=kv.device)
    )

    half_rd = rotary_dim // 2
    tail_len = D - 2 * half_rd
    block_d = max(triton.next_power_of_2(D), 16)
    block_r = max(triton.next_power_of_2(max(half_rd, 1)), 1)
    block_tail = max(triton.next_power_of_2(max(tail_len, 1)), 1)
    mid = torch.empty((L, T, H, D), dtype=torch.float32, device=kv.device)

    _k_norm_fp32_kernel[(T * L * H,)](
        kv,
        w,
        eps_t,
        mid,
        T,
        L,
        H,
        D,
        kv.stride(0),
        kv.stride(1),
        w.stride(0),
        eps_t.stride(0),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    _k_rope_cast_kernel[(T * L * H,)](
        mid,
        cos_sin_cache,
        positions,
        k_out,
        T,
        L,
        H,
        D,
        rotary_dim,
        half_rd,
        tail_len,
        cos_sin_cache.stride(0),
        BLOCK_R=block_r,
        BLOCK_TAIL=block_tail,
        HAS_ROPE=half_rd > 0,
        HAS_TAIL=tail_len > 0,
        num_warps=4,
        num_stages=1,
    )
    _v_copy_kernel[(T * L,)](
        kv,
        v_out,
        T,
        L,
        kv_size,
        kv.stride(0),
        kv.stride(1),
        BLOCK_V=triton.next_power_of_2(kv_size),
        num_warps=4,
        num_stages=1,
    )
    return k_out, v_out


__all__ = ["fused_norm_rope_stacked"]

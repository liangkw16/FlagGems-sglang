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

# Ascend vendor: one program per (t, l, h) row with 1D [BLOCK_D] tiles.
# The e1 generic left the largest case mismatched on this backend
# (2D [HEADS_TILE, BLOCK_D] tile + axis=1 reduction); the row form
# matches the platform-proven T51 ascend structure (tl.rsqrt).

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row_kernel(
    kv_ptr,
    w_ptr,
    eps_ptr,
    cos_sin_ptr,
    pos_ptr,
    k_out_ptr,
    v_out_ptr,
    num_layers,
    kv_size,
    num_heads,
    head_dim,
    rotary_dim,
    half_rd,
    kv_stride_t,
    kv_stride_l,
    cs_stride,
    w_stride_l,
    eps_stride,
    kout_stride_l,
    kout_stride_t,
    vout_stride_l,
    vout_stride_t,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % num_heads
    tl_idx = row // num_heads
    l = tl_idx % num_layers
    t = tl_idx // num_layers

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    kv_base = t * kv_stride_t + l * kv_stride_l + h * head_dim

    k = tl.load(kv_ptr + kv_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    var = tl.sum(k * k, axis=0) / head_dim
    eps = tl.load(eps_ptr + l * eps_stride).to(tl.float32)
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + l * w_stride_l + offs_d, mask=d_mask, other=1.0).to(tl.float32)
    k_normed = k * inv_rms * w

    out_base = (
        l * kout_stride_l + t * kout_stride_t + h * head_dim + offs_d
    )
    if HAS_ROPE:
        tl.store(
            k_out_ptr + out_base,
            k_normed.to(k_out_ptr.dtype.element_ty),
            mask=d_mask & (offs_d >= rotary_dim),
        )

        r = tl.arange(0, BLOCK_R)
        r_mask = r < half_rd
        pos = tl.load(pos_ptr + t)
        cos = tl.load(cos_sin_ptr + pos * cs_stride + r, mask=r_mask, other=0.0).to(
            tl.float32
        )
        sin = tl.load(
            cos_sin_ptr + pos * cs_stride + r + half_rd, mask=r_mask, other=0.0
        ).to(tl.float32)
        w_r1 = tl.load(w_ptr + l * w_stride_l + r, mask=r_mask, other=1.0).to(tl.float32)
        w_r2 = tl.load(
            w_ptr + l * w_stride_l + r + half_rd, mask=r_mask, other=1.0
        ).to(tl.float32)
        k1 = tl.load(kv_ptr + kv_base + r, mask=r_mask, other=0.0).to(tl.float32)
        k2 = tl.load(kv_ptr + kv_base + r + half_rd, mask=r_mask, other=0.0).to(
            tl.float32
        )
        k1 = k1 * inv_rms * w_r1
        k2 = k2 * inv_rms * w_r2
        rot1 = k1 * cos - k2 * sin
        rot2 = k2 * cos + k1 * sin
        out_ty = k_out_ptr.dtype.element_ty
        r_out = l * kout_stride_l + t * kout_stride_t + h * head_dim + r
        tl.store(k_out_ptr + r_out, rot1.to(out_ty), mask=r_mask)
        tl.store(k_out_ptr + r_out + half_rd, rot2.to(out_ty), mask=r_mask)
    else:
        tl.store(
            k_out_ptr + out_base,
            k_normed.to(k_out_ptr.dtype.element_ty),
            mask=d_mask,
        )

    v = tl.load(kv_ptr + kv_base + kv_size + offs_d, mask=d_mask, other=0.0)
    tl.store(
        v_out_ptr + l * vout_stride_l + t * vout_stride_t + h * head_dim + offs_d,
        v,
        mask=d_mask,
    )


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
    block_d = max(triton.next_power_of_2(D), 16)
    block_r = max(triton.next_power_of_2(max(half_rd, 1)), 1)

    _norm_rope_row_kernel[(T * L * H,)](
        kv,
        w,
        eps_t,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        L,
        kv_size,
        H,
        D,
        rotary_dim,
        half_rd,
        kv.stride(0),
        kv.stride(1),
        cos_sin_cache.stride(0),
        w.stride(0),
        eps_t.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        HAS_ROPE=half_rd > 0,
        num_warps=4,
        num_stages=1,
    )
    return k_out, v_out


__all__ = ["fused_norm_rope_stacked"]

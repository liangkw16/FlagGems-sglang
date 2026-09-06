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

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_stacked_kernel(
    kv_ptr,
    w_ptr,
    eps_ptr,
    cos_sin_ptr,
    pos_ptr,
    k_out_ptr,
    v_out_ptr,
    T,
    kv_size,
    num_kv_heads,
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
    HEADS_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    kv_stride_t = tl.cast(kv_stride_t, tl.int64)
    kv_stride_l = tl.cast(kv_stride_l, tl.int64)
    kout_stride_l = tl.cast(kout_stride_l, tl.int64)
    kout_stride_t = tl.cast(kout_stride_t, tl.int64)
    vout_stride_l = tl.cast(vout_stride_l, tl.int64)
    vout_stride_t = tl.cast(vout_stride_t, tl.int64)

    t = tl.program_id(0)
    l = tl.program_id(1)
    head_tile = tl.program_id(2)

    offs_h = head_tile * HEADS_TILE + tl.arange(0, HEADS_TILE)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    h_mask = offs_h < num_kv_heads
    row_base = t * kv_stride_t + l * kv_stride_l + offs_h[:, None] * head_dim + offs_d[None, :]
    hd_mask = h_mask[:, None] & d_mask[None, :]

    k_base = row_base
    v_base = row_base + kv_size

    # --- K: RMSNorm (fp32, per-head over D) ---
    k = tl.load(kv_ptr + k_base, mask=hd_mask, other=0.0).to(tl.float32)
    var = tl.sum(k * k, axis=1) / head_dim
    eps = tl.load(eps_ptr + l * eps_stride).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + l * w_stride_l + offs_d, mask=d_mask, other=1.0).to(tl.float32)
    k_normed = k * inv_rms[:, None] * w[None, :]

    out_base_l = l * kout_stride_l + t * kout_stride_t + offs_h[:, None] * head_dim + offs_d[None, :]

    # Tail (d >= rotary_dim) passes through normalized values; the rotary
    # region is written by the paired stores below, so mask it out here.
    if HAS_ROPE:
        tl.store(
            k_out_ptr + out_base_l,
            k_normed.to(k_out_ptr.dtype.element_ty),
            mask=hd_mask & (offs_d[None, :] >= rotary_dim),
        )
    else:
        tl.store(
            k_out_ptr + out_base_l,
            k_normed.to(k_out_ptr.dtype.element_ty),
            mask=hd_mask,
        )

    # --- K: RoPE on first rotary_dim dims ---
    if HAS_ROPE:
        r = tl.arange(0, BLOCK_R)
        r_mask = r < half_rd
        pos = tl.load(pos_ptr + t).to(tl.int64)
        cos = tl.load(cos_sin_ptr + pos * cs_stride + r, mask=r_mask, other=0.0).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + pos * cs_stride + r + half_rd, mask=r_mask, other=0.0
        ).to(tl.float32)

        r_base = (
            t * kv_stride_t
            + l * kv_stride_l
            + offs_h[:, None] * head_dim
            + r[None, :]
        )
        w_r1 = tl.load(w_ptr + l * w_stride_l + r, mask=r_mask, other=1.0).to(tl.float32)
        w_r2 = tl.load(
            w_ptr + l * w_stride_l + r + half_rd, mask=r_mask, other=1.0
        ).to(tl.float32)
        hr_mask = h_mask[:, None] & r_mask[None, :]
        k1 = tl.load(kv_ptr + r_base, mask=hr_mask, other=0.0).to(tl.float32)
        k2 = tl.load(kv_ptr + r_base + half_rd, mask=hr_mask, other=0.0).to(tl.float32)
        k1 = k1 * inv_rms[:, None] * w_r1[None, :]
        k2 = k2 * inv_rms[:, None] * w_r2[None, :]

        rot1 = k1 * cos[None, :] - k2 * sin[None, :]
        rot2 = k2 * cos[None, :] + k1 * sin[None, :]

        r_out = l * kout_stride_l + t * kout_stride_t + offs_h[:, None] * head_dim + r[None, :]
        out_ty = k_out_ptr.dtype.element_ty
        tl.store(k_out_ptr + r_out, rot1.to(out_ty), mask=hr_mask)
        tl.store(k_out_ptr + r_out + half_rd, rot2.to(out_ty), mask=hr_mask)

    # --- V: pure transpose copy ---
    v = tl.load(kv_ptr + v_base, mask=hd_mask, other=0.0)
    tl.store(
        v_out_ptr + l * vout_stride_l + t * vout_stride_t + offs_h[:, None] * head_dim + offs_d[None, :],
        v,
        mask=hd_mask,
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
    eps_t = eps.contiguous().reshape(-1) if isinstance(eps, torch.Tensor) else torch.full(
        (L,), float(eps), dtype=torch.float32, device=kv.device
    )

    half_rd = rotary_dim // 2
    block_d = max(triton.next_power_of_2(D), 16)
    block_r = max(triton.next_power_of_2(max(half_rd, 1)), 1)
    heads_tile = 4

    grid = (T, L, triton.cdiv(H, heads_tile))
    _norm_rope_stacked_kernel[grid](
        kv,
        w,
        eps_t,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        T,
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
        HEADS_TILE=heads_tile,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        HAS_ROPE=half_rd > 0,
        num_warps=4,
        num_stages=2,
    )
    return k_out, v_out


__all__ = ["fused_norm_rope_stacked"]

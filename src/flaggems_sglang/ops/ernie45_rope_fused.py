# r4 water re-roll carrier of the e7r2 sub-12399 team-best bytes (evening window).\n# Copyright 2026 FlagOS Contributors
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

_MAX_GRID = 65535


@triton.jit
def _ernie_rope_kernel(
    q_ptr,
    k_ptr,
    cos_sin_ptr,
    tpos_ptr,
    hpos_ptr,
    wpos_ptr,
    q_out_ptr,
    k_out_ptr,
    num_tokens,
    n_qh,
    n_kh,
    head_size,
    rotary_dim,
    section_hw,
    half_rd,
    q_stride,
    k_stride,
    qo_stride,
    ko_stride,
    cos_sin_stride,
    HEADS_TILE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DO_Q: tl.constexpr,
):
    token = tl.program_id(0)
    head_tile = tl.program_id(1)

    # Position selection per rotary pair index r in [0, half_rd)
    r = tl.arange(0, BLOCK_R)
    r_mask = r < half_rd
    use_hw = r < section_hw
    use_h = (r % 2) == 0
    h_pos = tl.load(hpos_ptr + token)
    w_pos = tl.load(wpos_ptr + token)
    t_pos = tl.load(tpos_ptr + token)
    pos_hw = tl.where(use_h, h_pos, w_pos)
    pos = tl.where(use_hw, pos_hw, t_pos)

    cos = tl.load(
        cos_sin_ptr + pos * cos_sin_stride + r,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + pos * cos_sin_stride + r + half_rd,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)

    head_start = head_tile * HEADS_TILE
    head_offsets = head_start + tl.arange(0, HEADS_TILE)

    if DO_Q:
        qh_mask = head_offsets < n_qh
        # x1: [HEADS_TILE, BLOCK_R] = q[token, head*hs + r]
        x1 = tl.load(
            q_ptr + token * q_stride + head_offsets[:, None] * head_size + r[None, :],
            mask=qh_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            q_ptr
            + token * q_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            mask=qh_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        new1 = x1 * cos[None, :] - x2 * sin[None, :]
        new2 = x2 * cos[None, :] + x1 * sin[None, :]
        q_ty = q_out_ptr.dtype.element_ty
        tl.store(
            q_out_ptr
            + token * qo_stride
            + head_offsets[:, None] * head_size
            + r[None, :],
            new1.to(q_ty),
            mask=qh_mask[:, None] & r_mask[None, :],
        )
        tl.store(
            q_out_ptr
            + token * qo_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            new2.to(q_ty),
            mask=qh_mask[:, None] & r_mask[None, :],
        )
        # Pass-through: rotary_dim to head_size
        tail_off = tl.arange(0, 64)
        for t0 in range(rotary_dim, head_size, 64):
            tail = t0 + tail_off
            t_mask = tail < head_size
            v = tl.load(
                q_ptr
                + token * q_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                mask=qh_mask[:, None] & t_mask[None, :],
                other=0.0,
            )
            tl.store(
                q_out_ptr
                + token * qo_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                v,
                mask=qh_mask[:, None] & t_mask[None, :],
            )
    else:
        kh_mask = head_offsets < n_kh
        x1 = tl.load(
            k_ptr + token * k_stride + head_offsets[:, None] * head_size + r[None, :],
            mask=kh_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            k_ptr
            + token * k_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            mask=kh_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        new1 = x1 * cos[None, :] - x2 * sin[None, :]
        new2 = x2 * cos[None, :] + x1 * sin[None, :]
        k_ty = k_out_ptr.dtype.element_ty
        tl.store(
            k_out_ptr
            + token * ko_stride
            + head_offsets[:, None] * head_size
            + r[None, :],
            new1.to(k_ty),
            mask=kh_mask[:, None] & r_mask[None, :],
        )
        tl.store(
            k_out_ptr
            + token * ko_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            new2.to(k_ty),
            mask=kh_mask[:, None] & r_mask[None, :],
        )
        tail_off = tl.arange(0, 64)
        for t0 in range(rotary_dim, head_size, 64):
            tail = t0 + tail_off
            t_mask = tail < head_size
            v = tl.load(
                k_ptr
                + token * k_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                mask=kh_mask[:, None] & t_mask[None, :],
                other=0.0,
            )
            tl.store(
                k_out_ptr
                + token * ko_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                v,
                mask=kh_mask[:, None] & t_mask[None, :],
            )


def ernie45_rope_fused(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim
):
    num_tokens, q_dim = q.shape
    k_dim = k.shape[1]
    n_qh = q_dim // head_size
    n_kh = k_dim // head_size
    half_rd = rotary_dim // 2
    section_h, section_w, section_t = mrope_section
    section_hw = section_h + section_w

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if num_tokens == 0:
        return q_out, k_out

    q = q.contiguous()
    k = k.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    tpos = positions[0].contiguous()
    hpos = positions[1].contiguous()
    wpos = positions[2].contiguous()

    block_r = max(triton.next_power_of_2(half_rd), 2)
    heads_tile = 4

    # Launch Q and K as separate grid calls sharing cos/sin loads
    grid_q = (num_tokens, triton.cdiv(n_qh, heads_tile))
    _ernie_rope_kernel[grid_q](
        q,
        k,
        cos_sin_cache,
        tpos,
        hpos,
        wpos,
        q_out,
        k_out,
        num_tokens,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        section_hw,
        half_rd,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        HEADS_TILE=heads_tile,
        BLOCK_R=block_r,
        DO_Q=True,
        num_warps=4,
        num_stages=1,
    )
    grid_k = (num_tokens, triton.cdiv(n_kh, heads_tile))
    _ernie_rope_kernel[grid_k](
        q,
        k,
        cos_sin_cache,
        tpos,
        hpos,
        wpos,
        q_out,
        k_out,
        num_tokens,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        section_hw,
        half_rd,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        HEADS_TILE=heads_tile,
        BLOCK_R=block_r,
        DO_Q=False,
        num_warps=4,
        num_stages=1,
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]

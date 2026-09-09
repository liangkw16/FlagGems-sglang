# e6: merged Q/K launch with shared per-tile cos/sin and int32 positions.
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
def _ernie_rope_qk_kernel(
    q_ptr,
    k_ptr,
    cos_sin_ptr,
    tpos_ptr,
    hpos_ptr,
    wpos_ptr,
    q_out_ptr,
    k_out_ptr,
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
):
    token = tl.program_id(0)
    head_tile = tl.program_id(1)

    # Position selection per rotary pair index r in [0, half_rd).
    # positions arrive as int32 (cast in the wrapper): int64 scalar
    # chains are the documented Ascend elementwise hazard, and pos
    # values cannot exceed the cos/sin cache row count.
    r = tl.arange(0, BLOCK_R)
    r_mask = r < half_rd
    use_hw = r < section_hw
    use_h = (r % 2) == 0
    h_pos = tl.load(hpos_ptr + token)
    w_pos = tl.load(wpos_ptr + token)
    t_pos = tl.load(tpos_ptr + token)
    pos_hw = tl.where(use_h, h_pos, w_pos)
    pos = tl.where(use_hw, pos_hw, t_pos)

    # One cos/sin gather per program, shared by the q and k head tiles.
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

    head_offsets = head_tile * HEADS_TILE + tl.arange(0, HEADS_TILE)
    tail_off = tl.arange(0, 64)

    qh_mask = head_offsets < n_qh
    x1 = tl.load(
        q_ptr
        + token * q_stride
        + head_offsets[:, None] * head_size
        + r[None, :],
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

    kh_mask = head_offsets < n_kh
    y1 = tl.load(
        k_ptr
        + token * k_stride
        + head_offsets[:, None] * head_size
        + r[None, :],
        mask=kh_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    y2 = tl.load(
        k_ptr
        + token * k_stride
        + head_offsets[:, None] * head_size
        + (r + half_rd)[None, :],
        mask=kh_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    k_new1 = y1 * cos[None, :] - y2 * sin[None, :]
    k_new2 = y2 * cos[None, :] + y1 * sin[None, :]
    k_ty = k_out_ptr.dtype.element_ty
    tl.store(
        k_out_ptr
        + token * ko_stride
        + head_offsets[:, None] * head_size
        + r[None, :],
        k_new1.to(k_ty),
        mask=kh_mask[:, None] & r_mask[None, :],
    )
    tl.store(
        k_out_ptr
        + token * ko_stride
        + head_offsets[:, None] * head_size
        + (r + half_rd)[None, :],
        k_new2.to(k_ty),
        mask=kh_mask[:, None] & r_mask[None, :],
    )
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
    # int32 positions: int64 scalar math is a documented Ascend hazard
    # and pos values are bounded by the cache row count.
    tpos = positions[0].to(torch.int32).contiguous()
    hpos = positions[1].to(torch.int32).contiguous()
    wpos = positions[2].to(torch.int32).contiguous()

    block_r = max(triton.next_power_of_2(half_rd), 2)
    heads_tile = 4
    grid = (num_tokens, triton.cdiv(max(n_qh, n_kh), heads_tile))
    _ernie_rope_qk_kernel[grid](
        q,
        k,
        cos_sin_cache,
        tpos,
        hpos,
        wpos,
        q_out,
        k_out,
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
        num_warps=4,
        num_stages=1,
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]

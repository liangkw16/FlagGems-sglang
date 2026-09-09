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

# Enflame/Kunlunxin vendor: wrapper precomputes the [T, half_rd]
# position tensor with PyTorch (pure metadata, removes all branching
# from the kernel); the kernel is a straight-line gather + rotate-half
# + tail copy with no tl.where on vectors.

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_precomputed_pos_kernel(
    x_ptr,
    out_ptr,
    cos_sin_ptr,
    pos_ptr,
    n_heads,
    head_size,
    half_rd,
    num_tokens,
    x_stride,
    o_stride,
    pos_stride,
    cos_sin_stride,
    HEADS_TILE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Persistent form for GCU: the vendor guide states kernel launch
    # overhead is NOT hidden, so a GridDim above the hardware resource
    # count is pure scheduling cost, worst of all when each program does
    # little work -- which is exactly this kernel (one token x head tile).
    # The old grid was (T, cdiv(n_heads, HEADS_TILE)), i.e. 1e3-1e4
    # programs against 24 SIPs. Launch a SIP-sized grid instead and walk
    # the flattened (token, head_tile) space in-kernel; tl.range carries
    # num_stages so the loop can pingpong.
    head_tiles = tl.cdiv(n_heads, HEADS_TILE)
    total_tiles = num_tokens * head_tiles
    r = tl.arange(0, BLOCK_R)
    r_mask = r < half_rd
    tail_off = tl.arange(0, 64)
    for tile in tl.range(
        tl.program_id(0), total_tiles, tl.num_programs(0), num_stages=3
    ):
        token = tile // head_tiles
        head_tile = tile - token * head_tiles
        head_offsets = head_tile * HEADS_TILE + tl.arange(0, HEADS_TILE)
        h_mask = head_offsets < n_heads

        # Precomputed position per (token, r) — pure gather, no branching
        pos = tl.load(pos_ptr + token * pos_stride + r, mask=r_mask, other=0)
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

        x1 = tl.load(
            x_ptr
            + token * x_stride
            + head_offsets[:, None] * head_size
            + r[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr
            + token * x_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        new1 = x1 * cos[None, :] - x2 * sin[None, :]
        new2 = x2 * cos[None, :] + x1 * sin[None, :]
        o_ty = out_ptr.dtype.element_ty
        tl.store(
            out_ptr
            + token * o_stride
            + head_offsets[:, None] * head_size
            + r[None, :],
            new1.to(o_ty),
            mask=h_mask[:, None] & r_mask[None, :],
        )
        tl.store(
            out_ptr
            + token * o_stride
            + head_offsets[:, None] * head_size
            + (r + half_rd)[None, :],
            new2.to(o_ty),
            mask=h_mask[:, None] & r_mask[None, :],
        )
        # Tail pass-through
        for t0 in range(2 * half_rd, head_size, 64):
            tail = t0 + tail_off
            t_mask = tail < head_size
            v = tl.load(
                x_ptr
                + token * x_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                mask=h_mask[:, None] & t_mask[None, :],
                other=0.0,
            )
            tl.store(
                out_ptr
                + token * o_stride
                + head_offsets[:, None] * head_size
                + tail[None, :],
                v,
                mask=h_mask[:, None] & t_mask[None, :],
            )


def _compute_positions(positions, mrope_section, half_rd, device):
    """Precompute [T, half_rd] position tensor (pure metadata)."""
    section_h, section_w, _ = mrope_section
    section_hw = section_h + section_w
    ridx = torch.arange(half_rd, device=device)
    use_hw = (ridx < section_hw).unsqueeze(0)  # [1, half_rd]
    use_h = ((ridx & 1) == 0).unsqueeze(0)
    hpos = positions[1].unsqueeze(1)  # [T, 1]
    wpos = positions[2].unsqueeze(1)
    tpos = positions[0].unsqueeze(1)
    pos_hw = torch.where(use_h, hpos, wpos)  # [T, half_rd]
    pos = torch.where(use_hw, pos_hw, tpos)  # [T, half_rd]
    return pos.to(torch.int32).contiguous()


def _apply_rope_precomputed(x, cos_sin_cache, pos, head_size, rotary_dim, heads_tile=4):
    T, x_dim = x.shape
    n_h = x_dim // head_size
    half_rd = rotary_dim // 2
    out = torch.empty_like(x)
    if T * x_dim == 0:
        return out
    block_r = max(triton.next_power_of_2(half_rd), 2)
    # GCU300 has 24 SIPs (2 die x 12); the vendor guide's recommended
    # GridDim is 6 at num_warps=4 and 24 at num_warps=8. Cap the grid at
    # the SIP count and let the persistent loop cover the rest, instead of
    # launching one program per (token, head_tile). num_warps is left to
    # the backend default: pinning it has been the wrong call on this chip
    # twice before (T19-E5, T51-E5).
    total_tiles = T * triton.cdiv(n_h, heads_tile)
    grid = (min(total_tiles, 24),)
    _rope_precomputed_pos_kernel[grid](
        x,
        out,
        cos_sin_cache,
        pos,
        n_h,
        head_size,
        half_rd,
        T,
        x.stride(0),
        out.stride(0),
        pos.stride(0),
        cos_sin_cache.stride(0),
        HEADS_TILE=heads_tile,
        BLOCK_R=block_r,
    )
    return out


def ernie45_rope_fused(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim
):
    half_rd = rotary_dim // 2
    pos = _compute_positions(positions, mrope_section, half_rd, q.device)
    q = q.contiguous()
    k = k.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    q_out = _apply_rope_precomputed(q, cos_sin_cache, pos, head_size, rotary_dim)
    k_out = _apply_rope_precomputed(k, cos_sin_cache, pos, head_size, rotary_dim)
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]

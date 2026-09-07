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

# Kunlunxin vendor (e3): per-pair micro programs following the official
# PR40 mrope_fused Kunlun work partition — grid=(tokens, head_groups,
# pairs), scalar position/cos/sin, a few head lanes, num_warps=1, and a
# separate tail-copy kernel. S0/e1/e2 kept a whole rotary-dim vector per
# program and all hit the uni_sram wall on this chip. The ERNIE4.5 axis
# rule (H/W alternating by pair parity, then the t position) replaces
# mrope_fused's contiguous T|H|W sections and is kept in scalar form.

import torch
import triton
import triton.language as tl

_BLOCK_HEADS = 16


@triton.jit
def _ernie_rope_pair_kernel(
    x_ptr,
    out_ptr,
    cos_sin_ptr,
    tpos_ptr,
    hpos_ptr,
    wpos_ptr,
    n_heads,
    head_size,
    half_rd,
    section_hw,
    x_stride,
    o_stride,
    cos_sin_stride,
    BLOCK_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1) * BLOCK_HEADS
    pair = tl.program_id(2)

    # ERNIE4.5 axis rule, scalar: pair < h+w alternates h/w by parity,
    # pair >= h+w uses the t position.
    use_hw = pair < section_hw
    use_h = (pair % 2) == 0
    h_pos = tl.load(hpos_ptr + token)
    w_pos = tl.load(wpos_ptr + token)
    t_pos = tl.load(tpos_ptr + token)
    pos = tl.where(use_hw, tl.where(use_h, h_pos, w_pos), t_pos)

    cache_base = cos_sin_ptr + pos.to(tl.int64) * cos_sin_stride
    cos = tl.load(cache_base + pair).to(tl.float32)
    sin = tl.load(cache_base + half_rd + pair).to(tl.float32)

    head = head_group + tl.arange(0, BLOCK_HEADS)
    head_mask = head < n_heads
    row_base = token.to(tl.int64) * x_stride + head * head_size
    first = tl.load(x_ptr + row_base + pair, mask=head_mask, other=0.0).to(
        tl.float32
    )
    second = tl.load(
        x_ptr + row_base + half_rd + pair, mask=head_mask, other=0.0
    ).to(tl.float32)
    o_ty = out_ptr.dtype.element_ty
    tl.store(
        out_ptr + row_base + pair,
        (first * cos - second * sin).to(o_ty),
        mask=head_mask,
    )
    tl.store(
        out_ptr + row_base + half_rd + pair,
        (second * cos + first * sin).to(o_ty),
        mask=head_mask,
    )


@triton.jit
def _ernie_rope_tail_kernel(
    x_ptr,
    out_ptr,
    n_heads,
    head_size,
    rotary_dim,
    x_stride,
    o_stride,
    BLOCK_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1) * BLOCK_HEADS
    tail = tl.program_id(2)
    head = head_group + tl.arange(0, BLOCK_HEADS)
    head_mask = head < n_heads
    offset = (
        token.to(tl.int64) * x_stride + head * head_size + rotary_dim + tail
    )
    tl.store(
        out_ptr + offset,
        tl.load(x_ptr + offset, mask=head_mask),
        mask=head_mask,
    )


def _apply_rope_pairs(
    x, cos_sin_cache, tpos, hpos, wpos, section_hw, head_size, rotary_dim
):
    num_tokens, x_dim = x.shape
    n_h = x_dim // head_size
    half_rd = rotary_dim // 2
    out = torch.empty_like(x)
    if out.numel() == 0:
        return out
    block_heads = _BLOCK_HEADS
    head_groups = triton.cdiv(n_h, block_heads)
    if half_rd > 0:
        grid = (num_tokens, head_groups, half_rd)
        _ernie_rope_pair_kernel[grid](
            x,
            out,
            cos_sin_cache,
            tpos,
            hpos,
            wpos,
            n_h,
            head_size,
            half_rd,
            section_hw,
            x.stride(0),
            out.stride(0),
            cos_sin_cache.stride(0),
            BLOCK_HEADS=block_heads,
            num_warps=1,
        )
    tail_size = head_size - rotary_dim
    if tail_size > 0:
        _ernie_rope_tail_kernel[(num_tokens, head_groups, tail_size)](
            x,
            out,
            n_h,
            head_size,
            rotary_dim,
            x.stride(0),
            out.stride(0),
            BLOCK_HEADS=block_heads,
            num_warps=1,
        )
    return out


def ernie45_rope_fused(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim
):
    section_h, section_w, _ = mrope_section
    section_hw = int(section_h) + int(section_w)
    q = q.contiguous()
    k = k.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    tpos = positions[0].contiguous()
    hpos = positions[1].contiguous()
    wpos = positions[2].contiguous()
    q_out = _apply_rope_pairs(
        q, cos_sin_cache, tpos, hpos, wpos, section_hw, head_size, rotary_dim
    )
    k_out = _apply_rope_pairs(
        k, cos_sin_cache, tpos, hpos, wpos, section_hw, head_size, rotary_dim
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]

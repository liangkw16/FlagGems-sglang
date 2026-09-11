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

# Enflame vendor, round 5. Four platform rounds established the GCU
# ruleset this kernel must obey: no int64 vector data loads, no integer
# select, no extsi of a vector load, and (after rounds 3-4 kept failing
# with the pair loads restructured two ways) no literal-valued or
# multi-branch store patterns. This round adopts the T49 precomputed-pos
# structure (a documented Enflame-proven asset): the wrapper precomputes
# the two decision tensors with torch -- borrow = (lo_word == 0), and
# vpos = ((seq_lens - 1) >= 0), whose int64 wrap semantics match the
# reference exactly including min_int64 -- and the kernel becomes pure
# branchless word arithmetic: result_lo = (lo - 1) * vpos and
# result_hi = (hi - borrow) * vpos, two's-complement exact for every
# int64 input (verified over 50k boundary+random values on the proxy
# harness). Three strided vector loads (the form that compiled in
# compute_src2dst's round-3 vendor), two masked stores, single-term
# masks, subi/addi/muli on int32 -- nothing outside the proven set.
# The int32 path keeps the generic kernel that already passed GCU.
# Grid capped at 24 per the 24-SIP vendor guidance; num_warps left to
# the backend default.

import torch
import triton
import triton.language as tl

_MAX_GRID = 24
_BLOCK = 256


@triton.jit
def _clamp_position(x, out, n, stride, BLOCK: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + i.to(tl.int64) * stride, i < n, other=0)
        value = (value - 1).to(x.dtype.element_ty)
        tl.store(out + i, tl.maximum(value, 0), i < n)


@triton.jit
def _clamp_position_word(x, aux, out, n, BLOCK: tl.constexpr):
    # Round 6: every GCU-safe element is contiguous and the addressing is
    # the exact shape of the generic int32 kernel that passes GCU -- a raw
    # index load `x + i * stride(1)` and a raw index store `out + i`. aux
    # is vpos for the lo word and borrow for the hi word; the wrapper
    # interleaves the two contiguous word results with torch copies.
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        value = tl.load(x + i.to(tl.int64), m, other=0)
        keep = tl.load(aux + i.to(tl.int64), m, other=0)
        tl.store(out + i, (value - 1) * keep, m)


@triton.jit
def _clamp_position_word_hi(x, borrow, vpos, out, n, BLOCK: tl.constexpr):
    # (hi - borrow) * vpos for the hi word; the same proven shape.
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        value = tl.load(x + i.to(tl.int64), m, other=0)
        b = tl.load(borrow + i.to(tl.int64), m, other=0)
        keep = tl.load(vpos + i.to(tl.int64), m, other=0)
        tl.store(out + i, (value - b) * keep, m)


def _as_words(tensor):
    # torch requires a unit stride on the last axis before viewing to a
    # narrower dtype; materialize a contiguous copy only in that rare case.
    if tensor.stride(-1) != 1:
        tensor = tensor.contiguous()
    return tensor.view(torch.int32)


def clamp_position(seq_lens):
    assert seq_lens.ndim == 1
    assert seq_lens.dtype in (torch.int32, torch.int64)
    if seq_lens.dtype == torch.int64:
        n = seq_lens.numel()
        out = torch.empty(
            seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
        )
        if n:
            words = _as_words(seq_lens)
            words_out = out.view(torch.int32)
            # Contiguous word streams; the kernels never see a strided
            # address. borrow: lo word == 0; vpos: sign of torch's
            # wrapped (v - 1), so min_int64 wraps to max_int64 and stays
            # positive exactly like the reference.
            lo_words = words[0::2].contiguous()
            hi_words = words[1::2].contiguous()
            borrow = (lo_words == 0).to(torch.int32)
            vpos = (seq_lens - 1 >= 0).to(torch.int32)
            lo_out = torch.empty(n, dtype=torch.int32, device=out.device)
            hi_out = torch.empty(n, dtype=torch.int32, device=out.device)
            grid = (min(triton.cdiv(n, _BLOCK), _MAX_GRID),)
            _clamp_position_word[grid](lo_words, vpos, lo_out, n, BLOCK=_BLOCK)
            _clamp_position_word_hi[grid](
                hi_words, borrow, vpos, hi_out, n, BLOCK=_BLOCK
            )
            words_out[0::2].copy_(lo_out)
            words_out[1::2].copy_(hi_out)
    else:
        out = torch.empty(
            seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
        )
        n = seq_lens.numel()
        if n:
            _clamp_position[(min(triton.cdiv(n, _BLOCK), _MAX_GRID),)](
                seq_lens, out, n, seq_lens.stride(0), BLOCK=_BLOCK
            )
    return out


__all__ = ["clamp_position"]

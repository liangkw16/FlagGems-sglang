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

# Enflame vendor, round 3 (submission 12903 feedback). Rounds 1-2 pinned
# the compile poison to int64 data paths: the generic fails exactly its
# int64 case and round 2's word-pair kernel (no i64 anywhere in the data
# path) still failed to compile with exactly one construct the
# same-round build_trtllm_mha_page_table vendor lacked -- an integer
# tl.where. Every Enflame-passed kernel in the corpus selects on floats
# only (apply_token_bitmask). This round removes the select entirely:
# the wrapper pre-fills the output with zeros (the clamp(min=0) branch
# materialized) and the kernel writes max(v-1, 0) via six masked stores
# whose masks are andi/cmpi chains -- all proven on GCU by round 2's
# compiled kernel and apply_token_bitmask. The positivity test needs no
# 64-bit compare or borrow arithmetic: v >= 1 is exactly
# (hi >= 0) & (max(lo, hi) != 0) (integer maxsi on i32 vectors passed
# GCU in the generic's int32 cases), and int64 wrap-around semantics
# for v == min_int64 (torch computes v-1 = max_int64) get their own
# store pair. int64 tensors are viewed as little-endian (lo, hi) int32
# word pairs in the wrapper (zero-copy reinterpret, full two's
# complement algebra preserved). Grid capped at 24 per the 24-SIP
# vendor guidance; num_warps left to the backend default.

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
def _clamp_position64(words, outw, n, BLOCK: tl.constexpr):
    # words/outw are little-endian int32 views of int64 tensors: element k
    # occupies words 2k (lo) and 2k+1 (hi). Non-positive inputs keep the
    # zero prefill. v >= 1 splits exactly into "lo != 0 under hi >= 0"
    # (store (lo-1, hi), no borrow) and "lo == 0" where v >= 1 collapses
    # to hi >= 1 (store (0xFFFFFFFF, hi-1), the borrow); v == min_int64
    # wraps like torch's int64 subtraction to max_int64.
    for block in range(
        tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)
    ):
        i = block * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        off = i.to(tl.int64) * 2
        lo = tl.load(words + off, m, other=0)
        hi = tl.load(words + off + 1, m, other=0)
        keep_lo = m & (hi >= 0) & (lo != 0)
        keep_hi = m & (hi >= 1) & (lo == 0)
        wrapped = m & (hi < -2147483647) & (lo == 0)
        tl.store(outw + off, lo - 1, keep_lo)
        tl.store(outw + off + 1, hi, keep_lo)
        tl.store(outw + off, -1, keep_hi)
        tl.store(outw + off + 1, hi - 1, keep_hi)
        tl.store(outw + off, -1, wrapped)
        tl.store(outw + off + 1, 2147483647, wrapped)


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
        out = torch.zeros(
            seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
        )
        n = seq_lens.numel()
        if n:
            _clamp_position64[
                (min(triton.cdiv(n, _BLOCK), _MAX_GRID),)
            ](_as_words(seq_lens), _as_words(out), n, BLOCK=_BLOCK)
    else:
        out = torch.empty(
            seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
        )
        n = seq_lens.numel()
        if n:
            _clamp_position[
                (min(triton.cdiv(n, _BLOCK), _MAX_GRID),)
            ](seq_lens, out, n, seq_lens.stride(0), BLOCK=_BLOCK)
    return out


__all__ = ["clamp_position"]

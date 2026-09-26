# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Task 107 tma_align_input_scale: transpose a per-token fp8 activation
# scale tensor into DeepGEMM's TMA column-major layout - M padded to a
# multiple of (16 / element_size) so each column is 16-byte aligned.
# The kernel copies input[m, k] into buffer[k, m_pad]; the wrapper
# returns buffer.t()[:M] whose values equal the input exactly.

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_pad_kernel(
    src,
    dst,
    M,
    K,
    m_pad,
    ss0,
    ds0,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    ks = pid.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    km = ks < K
    ms = tl.arange(0, BLOCK_M).to(tl.int64)
    for m0 in range(0, m_pad, BLOCK_M):
        mm = (m0 + ms) < M
        v = tl.load(
            src + (m0 + ms)[:, None] * ss0 + ks[None, :],
            mask=mm[:, None] & km[None, :],
            other=0.0,
        )
        tl.store(
            dst + ks[:, None] * ds0 + (m0 + ms)[None, :],
            tl.trans(v),
            mask=km[:, None] & mm[None, :],
        )


def tma_align_input_scale(input_scale):
    assert input_scale.dim() == 2
    m, k = input_scale.shape
    elem = input_scale.element_size()
    align = max(1, 16 // elem)
    m_pad = (m + align - 1) // align * align
    buf = torch.zeros((k, m_pad), dtype=input_scale.dtype, device=input_scale.device)
    if m and k:
        _transpose_pad_kernel[(triton.cdiv(k, 64),)](
            input_scale,
            buf,
            m,
            k,
            m_pad,
            input_scale.stride(0),
            buf.stride(0),
            BLOCK_K=64,
            BLOCK_M=max(16, min(1024, triton.next_power_of_2(m_pad))),
            num_warps=4,
        )
    return buf.t()[:m]


__all__ = ["tma_align_input_scale"]

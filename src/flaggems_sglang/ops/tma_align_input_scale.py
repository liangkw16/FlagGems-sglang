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
    ss1,
    ds0,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # one program per padded-m column: a 1D K-run load/store pair with
    # no transpose op and no 2D tile (the kunlunxin MLIR pipeline
    # rejected tl.trans and the shared-memory tiles blew limits)
    m = pid.to(tl.int64)
    ks = tl.arange(0, BLOCK_K).to(tl.int64)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + ks
        km = kk < K
        v = tl.load(src + m * ss0 + kk * ss1, mask=km, other=0.0)
        tl.store(dst + kk * ds0 + m, v, mask=km)


def tma_align_input_scale(input_scale):
    assert input_scale.dim() == 2
    m, k = input_scale.shape
    elem = input_scale.element_size()
    align = max(1, 16 // elem)
    m_pad = (m + align - 1) // align * align
    buf = torch.zeros((k, m_pad), dtype=input_scale.dtype, device=input_scale.device)
    if m and k:
        _transpose_pad_kernel[(m_pad,)](
            input_scale,
            buf,
            m,
            k,
            m_pad,
            input_scale.stride(0),
            input_scale.stride(1),
            buf.stride(0),
            BLOCK_K=min(1024, max(16, triton.next_power_of_2(k))),
            num_warps=4,
        )
    return buf.t()[:m]


__all__ = ["tma_align_input_scale"]

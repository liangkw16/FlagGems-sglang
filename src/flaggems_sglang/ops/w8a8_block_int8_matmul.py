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
def _w8a8_block_matmul_kernel(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_m,
    a_stride_k,
    b_stride_n,
    b_stride_k,
    as_stride_m,
    as_stride_k,
    bs_stride_n,
    bs_stride_k,
    c_stride_m,
    c_stride_n,
    group_n,
    group_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_stride_m = tl.cast(a_stride_m, tl.int64)
    a_stride_k = tl.cast(a_stride_k, tl.int64)
    b_stride_n = tl.cast(b_stride_n, tl.int64)
    b_stride_k = tl.cast(b_stride_k, tl.int64)
    c_stride_m = tl.cast(c_stride_m, tl.int64)
    c_stride_n = tl.cast(c_stride_n, tl.int64)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + (k_start + offs_k)[None, :] * a_stride_k,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + offs_n[None, :] * b_stride_n + (k_start + offs_k)[:, None] * b_stride_k,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_k = tl.dot(a, b, input_precision="ieee")
        k_group = k_start // group_k
        a_s = tl.load(
            as_ptr + offs_m * as_stride_m + k_group * as_stride_k,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        b_s = tl.load(
            bs_ptr + (offs_n // group_n) * bs_stride_n + k_group * bs_stride_k,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += acc_k * a_s[:, None] * b_s[None, :]

    tl.store(
        c_ptr + offs_m[:, None] * c_stride_m + offs_n[None, :] * c_stride_n,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _largest_pow2_leq(value, cap):
    v = min(int(value), cap)
    return 1 << (v.bit_length() - 1)


def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    M, K = A.shape
    N, _ = B.shape
    C = torch.empty((M, N), dtype=output_dtype, device=A.device)
    if M * N == 0 or K == 0:
        C.zero_()
        return C

    group_n, group_k = int(block_size[0]), int(block_size[1])
    # BLOCK_K must divide group_k so one k-iteration stays inside one scale
    # group; power-of-two block sizes make any pow2 <= group_k a divisor.
    block_m = 64
    block_n = min(_largest_pow2_leq(group_n, 64), triton.next_power_of_2(max(N, 16)))
    block_k = _largest_pow2_leq(group_k, 64)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _w8a8_block_matmul_kernel[grid](
        A,
        B,
        As,
        Bs,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        C.stride(0),
        C.stride(1),
        group_n,
        group_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return C


__all__ = ["w8a8_block_int8_matmul"]

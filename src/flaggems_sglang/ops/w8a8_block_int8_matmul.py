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
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_m,
    a_stride_k,
    b_stride_n,
    b_stride_k,
    as_stride_m,
    as_stride_k,
    bs_stride_n,
    bs_stride_k,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_stride_m = tl.cast(a_stride_m, tl.int64)
    a_stride_k = tl.cast(a_stride_k, tl.int64)
    b_stride_n = tl.cast(b_stride_n, tl.int64)
    b_stride_k = tl.cast(b_stride_k, tl.int64)
    as_stride_m = tl.cast(as_stride_m, tl.int64)
    as_stride_k = tl.cast(as_stride_k, tl.int64)
    bs_stride_n = tl.cast(bs_stride_n, tl.int64)
    bs_stride_k = tl.cast(bs_stride_k, tl.int64)

    # Each N tile belongs to one scale group, including ragged groups.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tiles_per_group = tl.cdiv(group_n, BLOCK_N)
    n_group = pid_n // tiles_per_group
    n_tile = pid_n % tiles_per_group
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = n_group * group_n + n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rows < M
    n_mask = (cols < N) & (cols < (n_group + 1) * group_n)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Reset the dot accumulator at every quantization-group boundary.
    for k_group in range(tl.cdiv(K, group_k)):
        group_sum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_tile in tl.static_range(tl.cdiv(group_k, BLOCK_K)):
            local_k = k_tile * BLOCK_K + offsets_k
            ks = k_group * group_k + local_k
            k_mask = (local_k < group_k) & (ks < K)
            a = tl.load(
                a_ptr + rows[:, None] * a_stride_m + ks[None, :] * a_stride_k,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + cols[None, :] * b_stride_n + ks[:, None] * b_stride_k,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0,
            ).to(tl.float32)
            group_sum = tl.dot(a, b, group_sum, input_precision="tf32")
        a_s = tl.load(
            as_ptr + rows * as_stride_m + k_group * as_stride_k,
            mask=m_mask,
            other=0,
        ).to(tl.float32)
        b_s = tl.load(bs_ptr + n_group * bs_stride_n + k_group * bs_stride_k)
        scales = a_s * b_s.to(tl.float32)
        accumulator += group_sum * scales[:, None]

    tl.store(
        c_ptr + rows[:, None].to(tl.int64) * N + cols[None, :],
        accumulator.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), dtype=output_dtype, device=A.device)
    if M * N == 0 or K == 0:
        C.zero_()
        return C
    group_n, group_k = map(int, block_size)
    block_m = 16 if M <= 16 else 32 if M <= 32 else 64
    tile_cap = 64
    block_n = 32 if M <= 32 else 64
    block_n = min(block_n, max(16, triton.next_power_of_2(group_n)))
    block_k = min(tile_cap, max(16, triton.next_power_of_2(group_k)))
    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, group_n) * triton.cdiv(group_n, block_n),
    )
    _w8a8_block_matmul_kernel[grid](
        A,
        B,
        As,
        Bs,
        C,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *As.stride(),
        *Bs.stride(),
        group_n,
        group_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8 if block_m * block_n >= 16384 else 4,
        num_stages=2,
    )
    return C


__all__ = ["w8a8_block_int8_matmul"]

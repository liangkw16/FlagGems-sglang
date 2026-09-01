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

# Kunlunxin: materialize the route once, run only regular GEMMs in Triton,
# then restore the original row order. This follows public PR41 and T28 E11.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _regular_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_BLOCK_M = 32
_BLOCK_N = 32
_BLOCK_K = 32
_GROUP_M = 8


def _launch_gemm(a, b, c, input_dim, output_dim):
    segment_len = a.shape[0]
    if segment_len == 0:
        return

    grid = (
        triton.cdiv(segment_len, _BLOCK_M) * triton.cdiv(output_dim, _BLOCK_N),
    )
    _regular_gemm_kernel[grid](
        a,
        b,
        c,
        segment_len,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        N=output_dim,
        K=input_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        num_warps=4,
        num_stages=1,
    )


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    seq_len, input_dim = x.shape
    output_dim = weights.shape[1]
    num_segments = int(batch_info.bs)
    output = torch.zeros((seq_len, output_dim), dtype=x.dtype, device=x.device)
    if (
        output.numel() == 0
        or num_segments == 0
        or input_dim == 0
        or output_dim == 0
    ):
        return output

    permutation = batch_info.permutation
    if permutation is not None:
        route = permutation.to(dtype=torch.int64, device=x.device).contiguous()
        route_cpu = route.cpu()
        inverse_cpu = torch.empty_like(route_cpu)
        inverse_cpu[route_cpu] = torch.arange(seq_len, dtype=torch.int64)
        inverse_route = inverse_cpu.to(device=x.device)
        x_packed = x.index_select(0, route).contiguous().float()
    else:
        inverse_route = None
        x_packed = x.contiguous().float()

    weights_fp32 = weights.contiguous().float()
    output_packed = torch.zeros(
        (seq_len, output_dim), dtype=torch.float32, device=x.device
    )
    indptr = batch_info.seg_indptr.tolist()
    weight_indices = batch_info.weight_indices.tolist()

    for segment in range(num_segments):
        start = int(indptr[segment])
        end = int(indptr[segment + 1])
        if start == end:
            continue
        adapter = int(weight_indices[segment])
        _launch_gemm(
            x_packed[start:end],
            weights_fp32[adapter],
            output_packed[start:end],
            input_dim,
            output_dim,
        )

    if inverse_route is not None:
        output_packed = output_packed.index_select(0, inverse_route)
    return output_packed.to(x.dtype)


__all__ = ["sgemm_lora_a"]

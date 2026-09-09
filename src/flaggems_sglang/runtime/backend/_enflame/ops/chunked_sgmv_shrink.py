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

# Enflame vendor (e6: retry): route/materialize with framework index_select and a
# regular GEMM per segment (the generic metadata-indirect kernel fails on this
# backend).
#
# E7 (T12 mirror): fp32-ieee dot operands are the pathological GCU
# configuration (T12 batch-2: ieee-fp32 dot + small tile + low stages ran
# 0.116x there, native fp16 operands with 64/64 tiles + stages 2 ran
# 0.743x). Keep bf16/fp16 inputs in their native dtype for the dot with
# an fp32 accumulator - bf16 x bf16 products are exact in fp32, so the
# math matches the reference's fp32 GEMM within the low-precision
# tolerances. True-fp32 inputs stay on the ieee path.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _shrink_gemm_kernel(
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
    USE_INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    offs_m = pid // num_pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid % num_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        if not USE_INPUT_DTYPE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_BLOCK_M = 64
_BLOCK_N = 64


def _launch_gemm(a, b, c, output_width, rank):
    m = a.shape[0]
    if m == 0:
        return
    bk = min(triton.next_power_of_2(max(rank, 16)), 128)
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(output_width, _BLOCK_N),)
    _shrink_gemm_kernel[grid](
        a,
        b,
        c,
        m,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        N=output_width,
        K=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=bk,
        USE_INPUT_DTYPE=(a.dtype in (torch.float16, torch.bfloat16)),
        num_warps=4,
        num_stages=2,
    )


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    output = torch.zeros(S, N, dtype=x.dtype, device=x.device)
    if output.numel() == 0 or batch_info.bs == 0:
        return output

    seg_indptr = batch_info.seg_indptr.detach().cpu().tolist()
    weight_indices = batch_info.weight_indices.detach().cpu().tolist()
    permutation = batch_info.permutation

    adapter_segments = {}
    for b in range(batch_info.bs):
        start, end = seg_indptr[b], seg_indptr[b + 1]
        if start == end:
            continue
        w_idx = weight_indices[b]
        if w_idx < 0:
            continue
        adapter_segments.setdefault(w_idx, []).append((start, end))

    for w_idx, segments in adapter_segments.items():
        if len(segments) == 1:
            start, end = segments[0]
            rows = permutation[start:end].long()
        else:
            rows = torch.cat(
                [permutation[start:end] for start, end in segments]
            ).long()
        x_seg = x.index_select(0, rows)
        out_seg = torch.empty(
            len(rows), N, dtype=torch.float32, device=x.device
        )
        _launch_gemm(x_seg, weights[w_idx], out_seg, N, K)
        output.index_copy_(0, rows, out_seg.to(x.dtype))
    return output


__all__ = ["chunked_sgmv_shrink"]

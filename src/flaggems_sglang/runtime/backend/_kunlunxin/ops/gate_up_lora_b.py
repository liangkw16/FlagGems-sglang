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

# Kunlunxin Task 28 — layout-materialization approach (PR41 pattern).
# Stage 1: framework layout — route/pack x and base into contiguous fp32,
#           transpose weights to [num_lora, r, 2*output_dim] (KN layout).
# Stage 2: pure regular [M,K]x[K,N] Triton GEMM per segment per gate/up.
# Stage 3: framework inverse — index_select to restore original row order.

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Regular [M,K]x[K,N] GEMM kernel — Kunlun-proven conservative shape.
# K=rank and N=output_dim are constexpr; M is runtime (do_not_specialize).
# No permutation, no seg_indptr, no indirect loads.
# ---------------------------------------------------------------------------
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
    scaling,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        mask_b = mask_k[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = offs_m
    offs_cn = offs_n
    c_ptrs = (
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    base = tl.load(c_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    result = base + acc * scaling
    tl.store(c_ptrs, result, mask=mask_c)


_BLOCK_M = 32
_BLOCK_N = 32
_BLOCK_K = 32
_GROUP_M = 8
_NUM_WARPS = 4
_NUM_STAGES = 1


def _launch_gemm(a, b, c, scaling, rank, output_dim):
    M, K = a.shape
    assert K == rank

    if M == 0:
        return

    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(output_dim, _BLOCK_N),)
    _regular_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        float(scaling),
        N=output_dim,
        K=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )


def gate_up_lora_b(x, gate_up_lora_b, batch_info, output_dim, base_output):
    rank = gate_up_lora_b.shape[-1]  # weights: [num_lora, 2*output_dim, r]

    if x.shape[1] != 2 * rank:
        raise ValueError("x width must equal 2 * rank")

    bs = batch_info.bs
    S = base_output.shape[0]

    if S == 0 or bs == 0 or output_dim <= 0 or rank == 0:
        return base_output.contiguous().clone()

    # ------------------------------------------------------------------
    # Stage 1 — Layout materialization (framework ops only, done once)
    # ------------------------------------------------------------------
    permutation = batch_info.permutation

    if permutation is not None:
        route = permutation.to(dtype=torch.int64, device=x.device)

        route_cpu = route.cpu()
        inv_cpu = torch.empty_like(route_cpu)
        inv_cpu[route_cpu] = torch.arange(S, dtype=torch.int64)
        inverse_route = inv_cpu.to(device=x.device)

        x_packed = x.index_select(0, route).contiguous().float()
        base_packed = base_output.index_select(0, route).contiguous().float()
    else:
        route = None
        inverse_route = None
        x_packed = x.contiguous().float()
        base_packed = base_output.contiguous().clone().float()

    # Materialize gate/up input views once globally: [S, r] each, contiguous fp32
    x_gate = x_packed[:, :rank].contiguous()  # [S, r]
    x_up = x_packed[:, rank:].contiguous()  # [S, r]

    # Transpose weights once: [num_lora, 2*output_dim, r] -> [num_lora, r, 2*output_dim]
    weights_kn = gate_up_lora_b.transpose(1, 2).contiguous().float()

    # Materialize gate/up weight views once globally: [num_lora, r, output_dim] each
    weights_gate_kn = weights_kn[:, :, :output_dim].contiguous()
    weights_up_kn = weights_kn[:, :, output_dim:].contiguous()

    # ------------------------------------------------------------------
    # Host-resolved segment metadata
    # ------------------------------------------------------------------
    indptr = batch_info.seg_indptr.tolist()
    weight_indices = batch_info.weight_indices.tolist()
    lora_ranks = batch_info.lora_ranks.tolist()
    scalings = batch_info.scalings.tolist()

    # ------------------------------------------------------------------
    # Stage 2 — Pure regular Triton GEMM per segment, gate and up separate
    # ------------------------------------------------------------------
    for b in range(bs):
        start = int(indptr[b])
        end = int(indptr[b + 1])
        length = end - start
        if length <= 0:
            continue

        wi = int(weight_indices[b])
        if int(lora_ranks[wi]) == 0:
            continue

        sc = float(scalings[wi])

        # Narrow views into the globally-materialized contiguous tensors
        a_gate = x_gate[
            start:end
        ]  # [length, r], stride from contiguous parent
        a_up = x_up[start:end]  # [length, r], stride from contiguous parent

        b_gate = weights_gate_kn[wi]  # [r, output_dim], contiguous
        b_up = weights_up_kn[wi]  # [r, output_dim], contiguous

        # C slices: row stride is 2*output_dim (base_packed is [S, 2*output_dim])
        c_gate = base_packed[start:end, :output_dim]
        c_up = base_packed[start:end, output_dim:]

        _launch_gemm(a_gate, b_gate, c_gate, sc, rank, output_dim)
        _launch_gemm(a_up, b_up, c_up, sc, rank, output_dim)

    # ------------------------------------------------------------------
    # Stage 3 — Inverse restore + dtype cast
    # ------------------------------------------------------------------
    if inverse_route is not None:
        result_fp32 = base_packed.index_select(0, inverse_route)
    else:
        result_fp32 = base_packed

    return result_fp32.to(base_output.dtype)


__all__ = ["gate_up_lora_b"]

# water-sample carrier r1 (2026-09-02): bytes identical to e11-b40e5aa team best
# carrier re-issued as e13 after stale_after_upload guard (upload produced no submission, quota unchanged)
# water-sample carrier r2 (2026-09-02)
# water-sample carrier r3 (2026-09-03)
# re-issue after stale_after_upload guard (2026-09-03): no submission was created
# water-sample carrier r5 (2026-09-03 night)

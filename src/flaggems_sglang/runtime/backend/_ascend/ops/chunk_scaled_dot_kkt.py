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

# Ascend vendor: FLA PR #1023 persistent structure (2026-08-19,
# hardware-tested on 910B3/CANN 9.0). Grid = physical AI cores,
# tl.range task stride, each task = one (chunk, batch*head) with BT=64
# whole tile, UB-computed BK (not hardcoded 128), beta/g pre-transposed
# to [H,B,T] contiguous, direct pointer arithmetic (no block_ptr).
# Peak live memory: b_A[BT,BT] + b_k[BT,BK]*2, multiplier 5.0, safety
# 0.85, budget 1572864 bits -> BK=32 safe for BT=64.

import torch
import triton
import triton.language as tl

_NUM_CORE = 32  # conservative Ascend AI core count
_SAFETY = 0.85
_MEM_MULT = 5.0
_UB_BYTES = 1572864 // 8  # 196608 bytes


def _ub_safe_bk(bt, k_size):
    """UB-safe BK for BT x BT accumulator + 2 x BT x BK operands."""
    budget = int(_UB_BYTES * _SAFETY)
    acc = bt * bt * 4
    avail = budget - acc
    if avail <= 0:
        return 16
    bk = avail // int(_MEM_MULT * bt * 4)
    bk = min(bk, triton.next_power_of_2(k_size))
    # round down to nearest power of 2 (tl.arange requirement)
    if bk > 16:
        bk = 1 << (bk.bit_length() - 1)
    return max(16, min(128, bk))


@triton.jit(do_not_specialize=["T", "B", "bh_step", "task_num", "num_core"])
def _kkt_fla_kernel(
    k_ptr,
    g_ptr,
    beta_ptr,
    A_ptr,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    bh_step,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    T = T.to(tl.int64)
    B = B.to(tl.int64)
    bt_stride = B * T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_i = task_id // bh_step
        i_bh = task_id % bh_step
        i_b = i_bh // HV
        i_h = i_bh % HV

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_t_i * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int64)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_local = eos - bos
        else:
            bos = i_b * T
            i_t = i_t_i.to(tl.int64)
            T_local = T

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T_local

        # beta/g are pre-transposed to [HV, B, T] contiguous
        p_beta = beta_ptr + i_h * bt_stride + bos + o_t
        b_beta = tl.load(p_beta, mask=m_t, other=0.0)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        # GQA: query head i_h maps to key head i_h // (HV // H)
        kg = i_h // (HV // H)
        for i_k in range(0, tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            p_k = k_ptr + (bos * H + kg) * K + o_t[:, None] * (H * K) + o_k[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
            b_A += tl.dot(b_k, tl.trans(b_k), input_precision="ieee")

        if USE_G:
            p_g = g_ptr + i_h * bt_stride + bos + o_t
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_g_diff = b_g[:, None] - b_g[None, :]
            # Task-mandated safe-exp: exponent <= 0 -> exp, else -> 0
            b_A *= tl.where(b_g_diff <= 0.0, tl.exp(b_g_diff), 0.0)

        b_A *= b_beta[:, None].to(tl.float32)
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0.0)

        p_A = (
            A_ptr
            + (bos * HV + i_h) * BT
            + o_t[:, None] * (BT * HV)
            + tl.arange(0, BT)[None, :]
        )
        tl.store(p_A, b_A.to(A_ptr.dtype.element_ty), mask=m_t[:, None])


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    batch, seqlen, num_k_heads, k_size = k.shape
    num_heads = beta.shape[-1]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seqlen % chunk_size:
        raise ValueError("seqlen must be divisible by chunk_size")
    if num_heads % num_k_heads:
        raise ValueError("num_heads must be divisible by num_k_heads")

    # Cap BT at 64: the [BT,BT] fp32 accumulator alone needs BT*BT*4
    # bytes; chunk_size=256 would need 256KB > the 192KB UB budget
    BT = min(chunk_size, 64)
    BK = _ub_safe_bk(BT, k_size)
    nchunks = seqlen // BT

    A = torch.zeros(
        (batch, seqlen, num_heads, BT),
        dtype=torch.float32,
        device=k.device,
    )
    if A.numel() == 0:
        return A

    # P1 fix: the kernel hardcodes contiguous [B,T,H,K] address math;
    # non-contiguous k would be silently wrong
    k = k.contiguous()
    use_g = g_cumsum is not None

    # Pre-transpose beta/g from [B, T, H] to [H, B, T] contiguous
    beta_t = beta.permute(2, 0, 1).contiguous()
    g_t = g_cumsum.permute(2, 0, 1).contiguous() if use_g else beta_t

    bh_step = batch * num_heads
    task_num = nchunks * bh_step
    _kkt_fla_kernel[(_NUM_CORE,)](
        k,
        g_t,
        beta_t,
        A,
        None,  # cu_seqlens (fixed-batch only)
        None,  # chunk_indices
        seqlen,
        batch,
        bh_step,
        task_num,
        _NUM_CORE,
        H=num_k_heads,
        HV=num_heads,
        K=k_size,
        BT=BT,
        BK=BK,
        IS_VARLEN=False,
        USE_G=use_g,
        num_warps=4,
        num_stages=1,
    )
    return A


__all__ = ["chunk_scaled_dot_kkt"]

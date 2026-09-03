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

_BLOCK_V = 1024


@triton.jit
def _chain_accept_kernel(
    candidates_ptr,
    retrive_index_ptr,
    uniform_samples_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    accept_count_ptr,
    seqlen,
    vocab_size,
    t_stride_row,
    t_stride_vocab,
    d_stride_row,
    d_stride_vocab,
):
    # The reference's cur_row is always s-1 at step s, so each acceptance
    # test is independent: accept[s] = (coin*q)[rounded to dtype] < p.
    # The accept count is the leading run of Trues; everything stays
    # branchless (scalar selects + masked stores only).
    elem_ty = target_probs_ptr.dtype.element_ty
    t_stride_row = tl.cast(t_stride_row, tl.int64)
    t_stride_vocab = tl.cast(t_stride_vocab, tl.int64)
    d_stride_row = tl.cast(d_stride_row, tl.int64)
    d_stride_vocab = tl.cast(d_stride_vocab, tl.int64)
    b = tl.program_id(0)
    row_base = b * seqlen
    root = tl.load(retrive_index_ptr + row_base)
    tl.store(accept_index_ptr + row_base, root)

    k = tl.zeros((), dtype=tl.int32)
    for s in range(1, seqlen):
        draft_token = tl.load(candidates_ptr + row_base + s)
        p = tl.load(
            target_probs_ptr
            + (row_base + s - 1) * t_stride_row
            + draft_token * t_stride_vocab
        ).to(tl.float32)
        q = tl.load(
            draft_probs_ptr
            + (b * (seqlen - 1) + (s - 1)) * d_stride_row
            + draft_token * d_stride_vocab
        ).to(tl.float32)
        coin = tl.load(uniform_samples_ptr + b * (seqlen - 1) + (s - 1)).to(tl.float32)
        # coin*q in torch rounds to the input dtype before comparing;
        # replicate that rounding exactly.
        prod = (coin * q).to(elem_ty).to(tl.float32)
        accepted = prod < p
        active = k == s - 1
        take = active & accepted
        k += take.to(tl.int32)

        slot = tl.load(retrive_index_ptr + row_base + s - 1)
        tl.store(predicts_ptr + slot, draft_token, mask=take)

    tl.store(accept_token_num_ptr + b, k)
    tl.store(accept_count_ptr + b, k)
    for j in range(0, seqlen):
        v = tl.load(retrive_index_ptr + row_base + j)
        tl.store(
            accept_index_ptr + row_base + j,
            tl.where(j <= k, v, -1),
        )


@triton.jit
def _chain_final_sample_kernel(
    target_probs_ptr,
    draft_probs_ptr,
    uniform_final_ptr,
    retrive_index_ptr,
    predicts_ptr,
    accept_count_ptr,
    seqlen,
    vocab_size,
    t_stride_row,
    t_stride_vocab,
    d_stride_row,
    d_stride_vocab,
    BLOCK_V: tl.constexpr,
):
    # Final inverse-CDF draw per request. val elements are rounded to
    # the input dtype exactly like the reference (fp32 math, dtype
    # store), the running sum/prefixes are fp32 with serial
    # block-carry, and each prefix is rounded back to the dtype before
    # the > threshold comparison, matching torch cumsum's opmath
    # rounding granularity.
    elem_ty = target_probs_ptr.dtype.element_ty
    t_stride_row = tl.cast(t_stride_row, tl.int64)
    t_stride_vocab = tl.cast(t_stride_vocab, tl.int64)
    d_stride_row = tl.cast(d_stride_row, tl.int64)
    d_stride_vocab = tl.cast(d_stride_vocab, tl.int64)
    b = tl.program_id(0)
    k = tl.load(accept_count_ptr + b)
    all_accepted = k == seqlen - 1
    prob_row = (b * seqlen + k) * t_stride_row
    # draft_probs only has seqlen-1 rows per request.
    draft_row = (b * (seqlen - 1) + k) * d_stride_row

    running = tl.zeros((), dtype=tl.float32)
    first_idx = tl.zeros((), dtype=tl.int32) + vocab_size
    found = tl.zeros((), dtype=tl.int32)
    coin_final = tl.load(uniform_final_ptr + b).to(tl.float32)

    # Pass 1: total sum of the corrected distribution.
    for v_block in range(0, tl.cdiv(vocab_size, BLOCK_V)):
        offs = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        p = tl.load(
            target_probs_ptr + prob_row + offs * t_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            draft_probs_ptr + draft_row + offs * d_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.where(q != q, 0.0, q)
        corrected = tl.maximum(p - q, 0.0)
        val = tl.where(all_accepted, p, corrected)
        val = val.to(elem_ty).to(tl.float32)
        running += tl.sum(val, axis=0)
    norm_sum = running.to(elem_ty).to(tl.float32)
    target_u = (coin_final * norm_sum).to(elem_ty).to(tl.float32)

    # Pass 2: first prefix (dtype-rounded) strictly above target_u.
    carry = tl.zeros((), dtype=tl.float32)
    for v_block in range(0, tl.cdiv(vocab_size, BLOCK_V)):
        offs = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        p = tl.load(
            target_probs_ptr + prob_row + offs * t_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            draft_probs_ptr + draft_row + offs * d_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.where(q != q, 0.0, q)
        corrected = tl.maximum(p - q, 0.0)
        val = tl.where(all_accepted, p, corrected)
        val = val.to(elem_ty).to(tl.float32)
        prefixes = carry + tl.cumsum(val, axis=0)
        match = prefixes.to(elem_ty).to(tl.float32) > target_u
        block_min = tl.min(tl.where(mask & match, offs, vocab_size), axis=0)
        new_first = tl.where(found == 0, tl.minimum(first_idx, block_min), first_idx)
        found += (block_min < vocab_size).to(tl.int32)
        first_idx = new_first
        carry += tl.sum(val, axis=0)

    final_token = tl.where(first_idx < vocab_size, first_idx, vocab_size - 1)
    slot = tl.load(retrive_index_ptr + b * seqlen + k)
    tl.store(predicts_ptr + slot, final_token.to(predicts_ptr.dtype.element_ty))


def chain_speculative_sampling(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    batch, seqlen = candidates.shape
    vocab_size = target_probs.shape[-1]
    device = candidates.device
    predicts = torch.zeros(num_slots, dtype=candidates.dtype, device=device)
    accept_index = torch.full(
        (batch, seqlen), -1, dtype=retrive_index.dtype, device=device
    )
    accept_token_num = torch.zeros(batch, dtype=torch.int32, device=device)
    if batch == 0:
        return predicts, accept_index, accept_token_num
    candidates = candidates.contiguous()
    retrive_index = retrive_index.contiguous()
    uniform_samples = uniform_samples.contiguous()
    uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.contiguous()
    accept_count = torch.empty(batch, dtype=torch.int32, device=device)
    # seqlen == 1 leaves draft_probs empty; all_accepted is always True
    # so its values are never used, but loads must stay in bounds.
    draft_safe = draft_probs if draft_probs.numel() > 0 else target_probs
    _chain_accept_kernel[(batch,)](
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_safe,
        predicts,
        accept_index,
        accept_token_num,
        accept_count,
        seqlen,
        vocab_size,
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(1),
        draft_probs.stride(2),
        num_warps=1,
        num_stages=1,
    )
    _chain_final_sample_kernel[(batch,)](
        target_probs,
        draft_safe,
        uniform_samples_for_final_sampling,
        retrive_index,
        predicts,
        accept_count,
        seqlen,
        vocab_size,
        target_probs.stride(1),
        target_probs.stride(2),
        draft_safe.stride(1),
        draft_safe.stride(2),
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        num_stages=1,
    )
    return predicts, accept_index, accept_token_num


__all__ = ["chain_speculative_sampling"]

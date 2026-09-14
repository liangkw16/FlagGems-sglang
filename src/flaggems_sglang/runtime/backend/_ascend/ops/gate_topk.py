# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Ascend vendor: same rationale as the enflame variant - the generic's
# tl.topk / tl.bitonic_merge / tl.sort streaming form has no platform
# evidence on this stack (T70 e1 died in the exec-0ms crash family).
# Per-row iterative extraction from primitives the platform accepts
# (tl.max + tl.min(tl.where(...)), the T27-proven shape), live-lane
# mask so all-(-inf) rows cannot re-select a column, branchless +inf
# key for torch.topk's NaN-greatest ordering while the stored value
# reloads the original bits, smallest-column tie-break, and a
# grid-stride over rows so coreDim stays inside the 65535 limit for
# any batch. i32 addressing throughout; num_warps left at the default.

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_topk_rows(
    x_ptr,
    values_ptr,
    indices_ptr,
    rows,
    n_cols,
    K: tl.constexpr,
    N_PAD: tl.constexpr,
):
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        cols = tl.arange(0, N_PAD)
        live = cols < n_cols
        x = tl.load(
            x_ptr + row * n_cols + cols,
            mask=live,
            other=float("-inf"),
        ).to(tl.float32)
        # torch.topk orders NaN greatest, strictly above +inf; mapping
        # NaN to +inf alone would collide with a real +inf input, so
        # selection is two-tier and branch-free: NaN candidates always
        # beat non-NaN candidates, smallest column wins inside a tier,
        # and the stored value reloads the original bits so NaN
        # payloads survive exactly.
        key = tl.where(x != x, float("inf"), x)
        for j in tl.static_range(K):
            m = tl.max(tl.where(live, key, float("-inf")), axis=0)
            cand = live & (key == m)
            nan_cand = cand & (x != x)
            rank = tl.where(
                nan_cand, cols, tl.where(cand, cols + N_PAD, 2 * N_PAD)
            )
            rank_min = tl.min(rank, axis=0)
            col = tl.where(rank_min >= N_PAD, rank_min - N_PAD, rank_min)
            dst = row * K + j
            tl.store(values_ptr + dst, tl.load(x_ptr + row * n_cols + col))
            tl.store(indices_ptr + dst, col.to(tl.int32))
            live = live & (cols != col)


def gate_topk(x, k):
    assert x.is_contiguous() and x.ndim == 2
    assert x.numel() <= 2**31
    assert 1 <= k <= 32
    n_rows, n_cols = x.shape
    assert k <= n_cols
    values = torch.empty((n_rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((n_rows, k), dtype=torch.int32, device=x.device)
    if n_rows and n_cols:
        _gate_topk_rows[(min(n_rows, 65535),)](
            x,
            values,
            indices,
            n_rows,
            n_cols,
            K=k,
            N_PAD=triton.next_power_of_2(n_cols),
        )
    return values, indices


__all__ = ["gate_topk"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Kunlunxin vendor: the XMLIR Triton fork has no tl.topk (the generic's
# streaming kernel fails with AttributeError), so this variant falls back to
# a per-row iterative extraction built only from primitives the fork is
# known to accept - the same tl.max + tl.min(tl.where(...)) shape as the
# platform-proven T27 kunlun vendor. Selection uses a live-lane mask (never
# a value sentinel, so all-(-inf) rows cannot re-select a column) and picks
# NaN columns first with the smallest index, matching torch.topk's
# NaN-greatest ordering and the task's smaller-column tie-break.

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
        row64 = row.to(tl.int64)
        cols = tl.arange(0, N_PAD)
        live = cols < n_cols
        x = tl.load(
            x_ptr + row64 * n_cols + cols,
            mask=live,
            other=float("-inf"),
        ).to(tl.float32)
        for j in tl.static_range(K):
            nan_lanes = (x != x) & live
            if tl.sum(nan_lanes.to(tl.int32), axis=0) > 0:
                col = tl.min(tl.where(nan_lanes, cols, N_PAD), axis=0)
            else:
                m = tl.max(tl.where(live, x, float("-inf")), axis=0)
                col = tl.min(tl.where(live & (x == m), cols, N_PAD), axis=0)
            dst = row64 * K + j
            tl.store(values_ptr + dst, tl.load(x_ptr + row64 * n_cols + col))
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
            num_warps=4,
        )
    return values, indices


__all__ = ["gate_topk"]

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor: the generic's streaming kernel leans on tl.topk /
# tl.bitonic_merge / tl.sort, none of which have a proven lowering on
# the GCU stack (T70 e1 died in the exec-0ms crash family with the
# kunlun-only vendor selected). This variant keeps to the primitives
# the platform has accepted on GCU: per-row iterative extraction with
# tl.max + tl.min(tl.where(...)) (the T27-proven shape), a live-lane
# mask instead of a value sentinel so all-(-inf) rows cannot re-select
# a column, and NaN-first ordering via a branchless +inf key so no
# runtime scalar branch wraps any vector op (the GCU300 PassManager
# poison). Addressing stays in i32 end to end - the GCU i64 rejection
# is signature-level but i32 is free - and num_warps is left at the
# backend default (pinning it has repeatedly cost ~2x on GCU).

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
        # torch.topk orders NaN greatest; an +inf key selects NaN
        # columns first while the stored value still reloads the
        # original bits, so NaN payloads survive exactly.
        key = tl.where(x != x, float("inf"), x)
        for j in tl.static_range(K):
            m = tl.max(tl.where(live, key, float("-inf")), axis=0)
            col = tl.min(tl.where(live & (key == m), cols, N_PAD), axis=0)
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

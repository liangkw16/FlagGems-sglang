# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Ascend vendor for compute_position (e9 generic form with dual int32
# word stores): the e8/e10 platform rounds showed huawei's 4.3x gap to
# the field is insensitive to launch count and tile/loop shape, so the
# remaining suspect is the int64 vector store path itself - AscendVector
# widens poorly through 8-byte element stores. This vendor keeps the
# fused single-launch self-scan (chunked so any batch fits a fixed
# BLOCK_BS) and writes each int64 position as two int32 words through
# the standard-layout view (lo at 2j, hi at 2j+1), which halves the
# store width and keeps hi (=0 for all int31-domain values) on a
# separate controllable lane.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _fill_positions_dualword(
    positions_words,
    start_loc,
    prefix_lens,
    lens,
    batch,
    HAS_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    i = tl.program_id(0)
    start = tl.zeros((), dtype=tl.int64)
    for c0 in range(0, batch, BLOCK_BS):
        lanes = c0 + tl.arange(0, BLOCK_BS)
        seg = tl.load(lens + lanes, lanes < batch, other=0).to(tl.int64)
        start += tl.sum(tl.where(lanes < i, seg, 0), 0)
    tl.store(start_loc + i, start.to(tl.int32))
    seq_len = tl.load(lens + i)
    prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
    for off in range(0, seq_len, BLOCK):
        o = off + tl.arange(0, BLOCK)
        m = o < seq_len
        wide = prefix_len.to(tl.int64) + o
        lo = (wide & 0xFFFFFFFF).to(tl.int32)
        hi = (wide >> 32).to(tl.int32)
        idx = start + o
        tl.store(positions_words + 2 * idx, lo, mask=m)
        tl.store(positions_words + 2 * idx + 1, hi, mask=m)


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    assert extend_prefix_lens.ndim == extend_seq_lens.ndim == 1
    assert extend_prefix_lens.dtype == extend_seq_lens.dtype == torch.int32
    batch = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch
    device = extend_seq_lens.device
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=device
    )
    extend_start_loc = torch.empty(batch, dtype=torch.int32, device=device)
    if batch:
        _fill_positions_dualword[(batch,)](
            positions.view(torch.int32),
            extend_start_loc,
            extend_prefix_lens,
            extend_seq_lens,
            batch,
            HAS_PREFIX=has_prefix,
            BLOCK=1024,
            BLOCK_BS=1024,
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

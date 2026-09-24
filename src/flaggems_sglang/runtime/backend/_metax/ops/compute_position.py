# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Metax vendor for compute_position: the e10 tiled form as a
# chip-scoped variant. The e10 platform round read muxi 847.7 (+20% vs
# the e9 loop form's 704.3) while tianshu paid -11% for the dead-tile
# self-scans - vendoring isolates the win to the chip that wants it.
# Structure: single launch, grid=(batch, tiles); dead tiles (first
# offset past the segment) exit after two scalar loads; each live tile
# derives its row's exclusive start from one masked int64 vector sum.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _fill_positions_tiled(
    positions, start_loc, prefix_lens, lens, batch,
    HAS_PREFIX: tl.constexpr, BLOCK: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    i = tl.program_id(0)
    t = tl.program_id(1)
    seq_len = tl.load(lens + i)
    if (t == 0) | (t * BLOCK < seq_len):
        lanes = tl.arange(0, BLOCK_BS)
        seg = tl.load(lens + lanes, lanes < batch, other=0).to(tl.int64)
        start = tl.sum(tl.where(lanes < i, seg, 0), 0)
        if t == 0:
            tl.store(start_loc + i, start.to(tl.int32))
        prefix_len = tl.load(prefix_lens + i) if HAS_PREFIX else 0
        for off in range(
            t * BLOCK, seq_len, tl.num_programs(1) * BLOCK
        ):
            o = off + tl.arange(0, BLOCK)
            tl.store(
                positions + start + o,
                prefix_len.to(tl.int64) + o,
                mask=o < seq_len,
            )


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
        tiles = max(
            1, min(triton.cdiv(extend_seq_lens_sum, 1024), 1024)
        )
        _fill_positions_tiled[(batch, tiles)](
            positions,
            extend_start_loc,
            extend_prefix_lens,
            extend_seq_lens,
            batch,
            HAS_PREFIX=has_prefix,
            BLOCK=1024,
            BLOCK_BS=triton.next_power_of_2(max(batch, 16)),
        )
    return positions, extend_start_loc


__all__ = ["compute_position"]

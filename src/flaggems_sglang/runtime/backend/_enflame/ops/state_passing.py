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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256
_MAX_GRID = 12


@triton.jit(do_not_specialize=["nchunks"])
def _state_passing_kernel(
    states_ptr,
    dA_cumsum_ptr,
    initial_states_ptr,
    out_ptr,
    final_states_ptr,
    total_tiles,
    nchunks,
    nheads,
    dim,
    tiles_per_head,
    states_stride_b,
    states_stride_c,
    states_stride_h,
    states_stride_d,
    dA_stride_b,
    dA_stride_h,
    dA_stride_c,
    dA_stride_l,
    init_stride_b,
    init_stride_h,
    init_stride_d,
    out_stride_b,
    out_stride_c,
    out_stride_h,
    out_stride_d,
    final_stride_b,
    final_stride_h,
    final_stride_d,
    length,
    HAS_INITIAL_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    states_stride_b = tl.cast(states_stride_b, tl.int64)
    states_stride_c = tl.cast(states_stride_c, tl.int64)
    states_stride_h = tl.cast(states_stride_h, tl.int64)
    states_stride_d = tl.cast(states_stride_d, tl.int64)
    dA_stride_b = tl.cast(dA_stride_b, tl.int64)
    dA_stride_h = tl.cast(dA_stride_h, tl.int64)
    dA_stride_c = tl.cast(dA_stride_c, tl.int64)
    dA_stride_l = tl.cast(dA_stride_l, tl.int64)
    init_stride_b = tl.cast(init_stride_b, tl.int64)
    init_stride_h = tl.cast(init_stride_h, tl.int64)
    init_stride_d = tl.cast(init_stride_d, tl.int64)
    out_stride_b = tl.cast(out_stride_b, tl.int64)
    out_stride_c = tl.cast(out_stride_c, tl.int64)
    out_stride_h = tl.cast(out_stride_h, tl.int64)
    out_stride_d = tl.cast(out_stride_d, tl.int64)
    final_stride_b = tl.cast(final_stride_b, tl.int64)
    final_stride_h = tl.cast(final_stride_h, tl.int64)
    final_stride_d = tl.cast(final_stride_d, tl.int64)

    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    dim_offsets = tl.arange(0, BLOCK_SIZE)

    for tile_id in range(pid, total_tiles, grid_stride):
        dim_tile = tile_id % tiles_per_head
        batch_head = tile_id // tiles_per_head
        head = batch_head % nheads
        batch = batch_head // nheads
        dim_indices = dim_tile * BLOCK_SIZE + dim_offsets
        dim_mask = dim_indices < dim

        if HAS_INITIAL_STATES:
            initial_base = batch * init_stride_b + head * init_stride_h
            current = tl.load(
                initial_states_ptr + initial_base + dim_indices * init_stride_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            current = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for chunk in range(0, nchunks):
            out_base = batch * out_stride_b + chunk * out_stride_c + head * out_stride_h
            out_dtype = out_ptr.dtype.element_ty
            tl.store(
                out_ptr + out_base + dim_indices * out_stride_d,
                current.to(out_dtype),
                mask=dim_mask,
            )

            dA_offset = (
                batch * dA_stride_b
                + head * dA_stride_h
                + chunk * dA_stride_c
                + (length - 1) * dA_stride_l
            )
            decay = tl.exp(tl.load(dA_cumsum_ptr + dA_offset).to(tl.float32))

            states_base = (
                batch * states_stride_b
                + chunk * states_stride_c
                + head * states_stride_h
            )
            state = tl.load(
                states_ptr + states_base + dim_indices * states_stride_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            current = current * decay + state

        final_base = batch * final_stride_b + head * final_stride_h
        tl.store(
            final_states_ptr + final_base + dim_indices * final_stride_d,
            current,
            mask=dim_mask,
        )


def state_passing(
    states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, nchunks, nheads, dim = states.shape
    assert dA_cumsum.shape[:3] == (batch, nheads, nchunks)
    length = dA_cumsum.shape[-1]
    assert length >= 1
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)

    out = torch.empty(
        states.shape,
        dtype=states.dtype,
        device=states.device,
    )
    if nchunks == 0 or batch == 0 or nheads == 0 or dim == 0:
        if initial_states is not None:
            final_states = initial_states.float().clone()
        else:
            final_states = torch.zeros(
                (batch, nheads, dim),
                dtype=torch.float32,
                device=states.device,
            )
        return out, final_states

    final_states = torch.empty(
        (batch, nheads, dim),
        dtype=torch.float32,
        device=states.device,
    )
    tiles_per_head = triton.cdiv(dim, _BLOCK_SIZE)
    total_tiles = batch * nheads * tiles_per_head
    grid = (min(total_tiles, _MAX_GRID),)

    if initial_states is None:
        initial_ptr = states
        initial_strides = (0, 0, 0)
    else:
        initial_ptr = initial_states
        initial_strides = initial_states.stride()

    _state_passing_kernel[grid](
        states,
        dA_cumsum,
        initial_ptr,
        out,
        final_states,
        total_tiles,
        nchunks,
        nheads,
        dim,
        tiles_per_head,
        *states.stride(),
        *dA_cumsum.stride(),
        *initial_strides,
        *out.stride(),
        *final_states.stride(),
        length,
        HAS_INITIAL_STATES=initial_states is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return out, final_states


__all__ = ["state_passing"]

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
_MAX_ROWS = 65535


@triton.jit
def _state_passing_step_kernel(
    states_ptr,
    dA_ptr,
    current_ptr,
    out_ptr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    dim_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = tl.program_id(1)
    mask = dim_offsets < DIM
    offsets = row * DIM + dim_offsets

    current = tl.load(current_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        out_ptr + offsets,
        current.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    decay = tl.exp(tl.load(dA_ptr + row).to(tl.float32))
    state = tl.load(states_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(current_ptr + offsets, current * decay + state, mask=mask)


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

    if nchunks == 0 or batch == 0 or nheads == 0 or dim == 0:
        out = torch.empty_like(states)
        if initial_states is not None:
            final_states = initial_states.float().clone()
        else:
            final_states = torch.zeros(
                (batch, nheads, dim),
                dtype=torch.float32,
                device=states.device,
            )
        return out, final_states

    rows = batch * nheads
    states_chunks = (
        states.permute(1, 0, 2, 3).contiguous().view(nchunks, rows, dim)
    )
    dA_chunks = (
        dA_cumsum[..., -1].permute(2, 0, 1).contiguous().view(nchunks, rows)
    )
    out_chunks = torch.empty_like(states_chunks)
    if initial_states is None:
        current = torch.zeros(
            (rows, dim), dtype=torch.float32, device=states.device
        )
    else:
        current = initial_states.float().clone(
            memory_format=torch.contiguous_format
        )
        current = current.view(rows, dim)

    for chunk in range(nchunks):
        row_start = 0
        while row_start < rows:
            row_stop = min(row_start + _MAX_ROWS, rows)
            grid = (triton.cdiv(dim, _BLOCK_SIZE), row_stop - row_start)
            _state_passing_step_kernel[grid](
                states_chunks[chunk, row_start:row_stop],
                dA_chunks[chunk, row_start:row_stop],
                current[row_start:row_stop],
                out_chunks[chunk, row_start:row_stop],
                DIM=dim,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=1,
            )
            row_start = row_stop

    out = out_chunks.view(nchunks, batch, nheads, dim)
    out = out.permute(1, 0, 2, 3).contiguous()
    return out, current.view(batch, nheads, dim)


__all__ = ["state_passing"]

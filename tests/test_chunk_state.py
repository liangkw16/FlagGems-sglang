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

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunk_state.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunk_state_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference(B, x, dt, dA_cumsum):
    batch, _, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    x_chunks = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_chunks = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_chunks = B_chunks.repeat_interleave(ratio, dim=3)
    dA_last = dA_cumsum[..., -1:].float()
    decay = torch.exp(dA_last - dA_cumsum.float())
    scale = (decay * dt.float()).permute(0, 2, 3, 1)
    B_scaled = B_chunks * scale.unsqueeze(-1)
    return torch.einsum("bcthp,bcthn->bchpn", x_chunks, B_scaled)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkStateTest(unittest.TestCase):
    def test_gqa_strided_tail_matches_reference(self):
        torch.manual_seed(20260824)
        batch, nchunks, chunk_size = 2, 3, 17
        seqlen = nchunks * chunk_size
        nheads, ngroups = 6, 2
        headdim, dstate = 19, 23

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                B_storage = torch.randn(
                    (batch, seqlen, ngroups, dstate * 2),
                    device="cuda",
                    dtype=dtype,
                )
                x_storage = torch.randn(
                    (batch, seqlen, nheads, headdim * 2),
                    device="cuda",
                    dtype=dtype,
                )
                dt_storage = torch.empty(
                    (batch, nheads, nchunks, chunk_size * 2),
                    device="cuda",
                    dtype=dtype,
                )
                dA_storage = torch.empty_like(dt_storage)
                B = B_storage[..., ::2]
                x = x_storage[..., 1::2]
                dt = dt_storage[..., ::2]
                dA_cumsum = dA_storage[..., 1::2]
                dt.copy_(torch.rand_like(dt) * 0.1)
                dA_cumsum.copy_(-torch.rand_like(dA_cumsum).cumsum(-1))
                originals = tuple(
                    tensor.clone() for tensor in (B, x, dt, dA_cumsum)
                )

                actual = MODULE.chunk_state(B, x, dt, dA_cumsum)
                expected = _reference(B, x, dt, dA_cumsum)

                self.assertEqual(
                    actual.shape,
                    (batch, nchunks, nheads, headdim, dstate),
                )
                self.assertEqual(actual.dtype, torch.float32)
                self.assertFalse(B.is_contiguous())
                self.assertFalse(x.is_contiguous())
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )
                for tensor, original in zip((B, x, dt, dA_cumsum), originals):
                    torch.testing.assert_close(
                        tensor, original, atol=0.0, rtol=0.0
                    )

    def test_empty_batch_returns_empty_fp32(self):
        B = torch.empty((0, 17, 1, 7), device="cuda", dtype=torch.float16)
        x = torch.empty((0, 17, 2, 5), device="cuda", dtype=torch.float16)
        dt = torch.empty((0, 2, 1, 17), device="cuda", dtype=torch.float16)
        dA_cumsum = torch.empty_like(dt)

        actual = MODULE.chunk_state(B, x, dt, dA_cumsum)

        self.assertEqual(actual.shape, (0, 1, 2, 5, 7))
        self.assertEqual(actual.dtype, torch.float32)

    def test_chunk_k_boundaries_with_fp32_scales(self):
        torch.manual_seed(20260824)
        batch, nchunks = 1, 2
        nheads, ngroups = 4, 2
        headdim, dstate = 33, 35

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for chunk_size in (1, 31, 32, 33, 63, 64, 65, 255, 256, 257):
                with self.subTest(dtype=dtype, chunk_size=chunk_size):
                    seqlen = nchunks * chunk_size
                    B = torch.randn(
                        (batch, seqlen, ngroups, dstate),
                        device="cuda",
                        dtype=dtype,
                    )
                    x = torch.randn(
                        (batch, seqlen, nheads, headdim),
                        device="cuda",
                        dtype=dtype,
                    )
                    dt = torch.rand(
                        (batch, nheads, nchunks, chunk_size),
                        device="cuda",
                        dtype=torch.float32,
                    ).mul_(0.1)
                    dA_cumsum = -(torch.rand_like(dt) * 0.01).cumsum(-1)

                    actual = MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )


if __name__ == "__main__":
    unittest.main()

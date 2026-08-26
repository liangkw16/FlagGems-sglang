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
BACKEND_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
)
ASCEND_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_ascend"
    / "ops"
    / "chunk_state.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module("chunk_state_module", MODULE_PATH)
ASCEND_MODULE = _load_module("chunk_state_ascend_module", ASCEND_MODULE_PATH)
ILUVATAR_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_iluvatar"
    / "ops"
    / "chunk_state.py"
)
ILUVATAR_MODULE = _load_module(
    "chunk_state_iluvatar_module", ILUVATAR_MODULE_PATH
)
ENFLAME_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_enflame"
    / "ops"
    / "chunk_state.py"
)
ENFLAME_MODULE = _load_module(
    "chunk_state_enflame_module", ENFLAME_MODULE_PATH
)
E6_MODULES = tuple(
    (
        vendor,
        _load_module(
            f"chunk_state_{vendor}_module",
            BACKEND_PATH / f"_{vendor}" / "ops" / "chunk_state.py",
        ),
    )
    for vendor in ("kunlunxin", "metax", "amd")
)


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

    def test_generic_lowprec_tensor_core_path_precision(self):
        torch.manual_seed(20260825)
        shapes = (
            (1, 2, 256, 4, 2, 64, 128),
            (1, 2, 64, 4, 2, 64, 128),
            (2, 3, 17, 6, 2, 65, 129),
        )
        for (
            batch,
            nchunks,
            chunk_size,
            nheads,
            ngroups,
            headdim,
            dstate,
        ) in shapes:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(
                    shape=(
                        batch,
                        nchunks,
                        chunk_size,
                        nheads,
                        headdim,
                        dstate,
                    ),
                    dtype=dtype,
                ):
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

    def test_generic_fp32_path_keeps_ieee_precision(self):
        torch.manual_seed(20260825)
        batch, nchunks, chunk_size = 2, 4, 64
        seqlen = nchunks * chunk_size
        nheads, ngroups = 4, 2
        headdim, dstate = 64, 128

        B = torch.randn(
            (batch, seqlen, ngroups, dstate),
            device="cuda",
            dtype=torch.float32,
        )
        x = torch.randn(
            (batch, seqlen, nheads, headdim),
            device="cuda",
            dtype=torch.float32,
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
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    def test_e6_fallback_vendors_match_reference(self):
        torch.manual_seed(20260826)
        batch, nchunks, nheads, ngroups = 1, 2, 4, 2
        headdim, dstate = 33, 35

        for dtype in (torch.float16, torch.bfloat16):
            for chunk_size in (64, 256):
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
                expected = _reference(B, x, dt, dA_cumsum)

                for vendor, module in E6_MODULES:
                    with self.subTest(
                        vendor=vendor,
                        dtype=dtype,
                        chunk_size=chunk_size,
                    ):
                        actual = module.chunk_state(B, x, dt, dA_cumsum)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(
                            actual, expected, atol=3e-2, rtol=3e-2
                        )

    def test_ascend_capped_grid_covers_multi_iteration_scale(self):
        torch.manual_seed(20260824)
        shapes = (
            (2, 128, 64, 8, 2, 64, 64),
            (2, 3, 17, 6, 2, 19, 23),
        )
        for (
            batch,
            nchunks,
            chunk_size,
            nheads,
            ngroups,
            headdim,
            dstate,
        ) in shapes:
            tiles = (headdim + 31) // 32 * ((dstate + 31) // 32)
            total = tiles * batch * nchunks * nheads
            for dtype in (torch.float32, torch.float16):
                with self.subTest(
                    shape=(
                        batch,
                        nchunks,
                        chunk_size,
                        nheads,
                        headdim,
                        dstate,
                    ),
                    dtype=dtype,
                    total_programs=total,
                ):
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

                    actual = ASCEND_MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )

                    iluvatar_actual = ILUVATAR_MODULE.chunk_state(
                        B, x, dt, dA_cumsum
                    )
                    torch.testing.assert_close(
                        iluvatar_actual, expected, atol=3e-2, rtol=3e-2
                    )
        tiles = (64 + 31) // 32 * ((64 + 31) // 32)
        self.assertGreater(tiles * 2 * 128 * 8, 4096)

    def test_ascend_cube_path_precision(self):
        torch.manual_seed(20260826)
        batch, nchunks, nheads, ngroups = 1, 2, 8, 2

        for dtype in (torch.float16, torch.bfloat16):
            for chunk_size, headdim, dstate in (
                (64, 128, 128),
                (128, 64, 128),
                (256, 64, 64),
            ):
                with self.subTest(
                    dtype=dtype,
                    chunk_size=chunk_size,
                    headdim=headdim,
                    dstate=dstate,
                ):
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

                    actual = ASCEND_MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )

    def test_iluvatar_plain_dot_chunk_boundary_precision(self):
        torch.manual_seed(20260824)
        batch, nchunks = 1, 2
        nheads, ngroups = 4, 2
        headdim, dstate = 33, 35

        for dtype in (torch.float32, torch.float16):
            for chunk_size in (63, 64, 255, 256, 257):
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

                    actual = ILUVATAR_MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )

    def test_iluvatar_vendor_strided_inputs(self):
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

                actual = ILUVATAR_MODULE.chunk_state(B, x, dt, dA_cumsum)
                expected = _reference(B, x, dt, dA_cumsum)

                self.assertFalse(B.is_contiguous())
                self.assertFalse(x.is_contiguous())
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_enflame_dot_config_precision(self):
        torch.manual_seed(20260824)
        batch, nchunks = 1, 2
        nheads, ngroups = 4, 2
        headdim, dstate = 33, 35

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for chunk_size in (63, 64, 127, 128, 255, 256, 257):
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

                    actual = ENFLAME_MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )

    def test_enflame_vendor_strided_inputs(self):
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

                actual = ENFLAME_MODULE.chunk_state(B, x, dt, dA_cumsum)
                expected = _reference(B, x, dt, dA_cumsum)

                self.assertFalse(B.is_contiguous())
                self.assertFalse(x.is_contiguous())
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_enflame_fold_covers_multi_iteration(self):
        torch.manual_seed(20260824)
        shapes = (
            (2, 32, 64, 8, 2, 64, 128),
            (2, 3, 17, 6, 2, 19, 23),
        )
        for (
            batch,
            nchunks,
            chunk_size,
            nheads,
            ngroups,
            headdim,
            dstate,
        ) in shapes:
            tiles = (headdim + 63) // 64 * ((dstate + 63) // 64)
            total = tiles * batch * nchunks * nheads
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    shape=(
                        batch,
                        nchunks,
                        chunk_size,
                        nheads,
                        headdim,
                        dstate,
                    ),
                    dtype=dtype,
                    total_programs=total,
                ):
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

                    actual = ENFLAME_MODULE.chunk_state(B, x, dt, dA_cumsum)
                    expected = _reference(B, x, dt, dA_cumsum)

                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-2, rtol=3e-2
                    )
        tiles = (64 + 63) // 64 * ((128 + 63) // 64)
        self.assertGreater(tiles * 2 * 32 * 8, 64)


if __name__ == "__main__":
    unittest.main()

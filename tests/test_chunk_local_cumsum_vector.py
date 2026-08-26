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

BACKEND_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module(
    "chunk_local_cumsum_vector_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunk_local_cumsum_vector.py",
)
ASCEND_MODULE = _load_module(
    "chunk_local_cumsum_vector_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "chunk_local_cumsum_vector.py",
)
KUNLUN_MODULE = _load_module(
    "chunk_local_cumsum_vector_kunlunxin_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "chunk_local_cumsum_vector.py",
)
ENFLAME_MODULE = _load_module(
    "chunk_local_cumsum_vector_enflame_module",
    BACKEND_ROOT / "_enflame" / "ops" / "chunk_local_cumsum_vector.py",
)


def reference(g, chunk_size, reverse=False, scale=None):
    batch, seqlen, nheads, state = g.shape
    chunks = seqlen // chunk_size
    values = g.float().view(batch, chunks, chunk_size, nheads, state)
    if reverse:
        values = values.flip(2)
    output = values.cumsum(2)
    if scale is not None:
        output = output * scale
    if reverse:
        output = output.flip(2)
    return output.reshape(batch, seqlen, nheads, state)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkLocalCumsumVectorTest(unittest.TestCase):
    def test_matches_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        cases = ((2, 30, 3, 5, 5), (1, 256, 8, 16, 64))
        for dtype, tolerance in tolerances.items():
            for shape in cases:
                batch, seqlen, nheads, state, chunk_size = shape
                for reverse in (False, True):
                    for scale in (None, -0.25, 2.0):
                        with self.subTest(
                            dtype=dtype,
                            shape=shape,
                            reverse=reverse,
                            scale=scale,
                        ):
                            base = torch.randn(
                                batch,
                                seqlen,
                                nheads,
                                state * 2,
                                generator=generator,
                                device="cuda",
                                dtype=dtype,
                            )
                            g = base[..., ::2]
                            original = g.clone()

                            actual = MODULE.chunk_local_cumsum_vector(
                                g, chunk_size, reverse, scale
                            )
                            expected = reference(g, chunk_size, reverse, scale)

                            self.assertEqual(actual.dtype, torch.float32)
                            self.assertEqual(actual.shape, g.shape)
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=tolerance,
                                rtol=tolerance,
                            )
                            torch.testing.assert_close(
                                g, original, atol=0.0, rtol=0.0
                            )

    def test_vendors_preserve_folded_cumsum(self):
        batch, seqlen, nheads, state_size, chunk_size = 1, 16384, 64, 64, 128
        feature_blocks = (nheads * state_size + 7) // 8
        total = feature_blocks * (seqlen // chunk_size) * batch
        self.assertGreater(total, 65535)
        g = torch.randn(
            (batch, seqlen, nheads, state_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = reference(g, chunk_size)

        for name, module in (
            ("ascend", ASCEND_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_local_cumsum_vector(g, chunk_size)
                torch.testing.assert_close(
                    actual, expected, atol=1.5e-2, rtol=1.5e-2
                )

    def test_empty_input(self):
        g = torch.empty((2, 0, 3, 5), device="cuda")

        actual = MODULE.chunk_local_cumsum_vector(g, 5)

        self.assertEqual(actual.shape, g.shape)
        self.assertEqual(actual.dtype, torch.float32)

    def test_dot_vendors_cover_non_power_of_two_chunk(self):
        g = torch.randn((1, 320, 64, 256), device="cuda", dtype=torch.float32)

        for reverse in (False, True):
            expected = reference(g, 5, reverse=reverse, scale=1.5)
            for name, module in (
                ("ascend", ASCEND_MODULE),
                ("kunlunxin", KUNLUN_MODULE),
                ("enflame", ENFLAME_MODULE),
            ):
                with self.subTest(module=name, reverse=reverse):
                    actual = module.chunk_local_cumsum_vector(
                        g, 5, reverse=reverse, scale=1.5
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=1e-4, rtol=1e-4
                    )

    def test_dot_vendors_cover_segmented_grid(self):
        batch, seqlen, nheads, state_size, chunk_size = 1, 128, 64, 256, 1
        feature_blocks = (nheads * state_size + 31) // 32
        total = feature_blocks * (seqlen // chunk_size) * batch
        self.assertGreater(total, 65535)
        g = torch.randn(
            (batch, seqlen, nheads, state_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = reference(g, chunk_size, reverse=True)

        for name, module in (
            ("ascend", ASCEND_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_local_cumsum_vector(
                    g, chunk_size, reverse=True
                )
                torch.testing.assert_close(
                    actual, expected, atol=1.5e-2, rtol=1.5e-2
                )

    def test_vendors_cover_large_grid(self):
        torch.manual_seed(20260824)
        batch, seqlen, nheads, state_size, chunk_size = 2, 8192, 32, 16, 64
        block_f = 8
        feature_blocks = (nheads * state_size + block_f - 1) // block_f
        total = feature_blocks * (seqlen // chunk_size) * batch
        self.assertGreater(total, 4096)

        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
            (torch.bfloat16, 1.5e-2),
        ):
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    g = torch.randn(
                        (batch, seqlen, nheads, state_size),
                        device="cuda",
                        dtype=dtype,
                    )
                    expected = reference(
                        g, chunk_size, reverse=reverse, scale=1.5
                    )
                    for name, module in (
                        ("generic", MODULE),
                        ("ascend", ASCEND_MODULE),
                        ("kunlunxin", KUNLUN_MODULE),
                        ("enflame", ENFLAME_MODULE),
                    ):
                        with self.subTest(module=name):
                            actual = module.chunk_local_cumsum_vector(
                                g, chunk_size, reverse=reverse, scale=1.5
                            )
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=tolerance,
                                rtol=tolerance,
                            )


if __name__ == "__main__":
    unittest.main()

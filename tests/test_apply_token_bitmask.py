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

ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module(
    "apply_token_bitmask_module",
    ROOT / "src/flaggems_sglang/ops/apply_token_bitmask.py",
)
VENDOR_MODULES = {
    "ascend": load_module(
        "apply_token_bitmask_ascend_module",
        ROOT
        / "src/flaggems_sglang/runtime/backend/_ascend/ops"
        / "apply_token_bitmask.py",
    ),
    "enflame": load_module(
        "apply_token_bitmask_enflame_module",
        ROOT
        / "src/flaggems_sglang/runtime/backend/_enflame/ops"
        / "apply_token_bitmask.py",
    ),
}


def reference(logits, bitmask):
    vocab_size = logits.shape[1]
    token = torch.arange(vocab_size, device=logits.device)
    word = token // 32
    bit = token % 32
    allowed = ((bitmask[:, word] >> bit) & 1) != 0
    return torch.where(allowed, logits, torch.full_like(logits, -float("inf")))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ApplyTokenBitmaskTest(unittest.TestCase):
    def test_word_and_block_boundaries_all_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for vocab_size in (31, 32, 33, 255, 256, 257):
                with self.subTest(dtype=dtype, vocab_size=vocab_size):
                    logits = (
                        torch.linspace(
                            -3,
                            3,
                            2 * vocab_size,
                            device="cuda",
                        )
                        .reshape(2, vocab_size)
                        .to(dtype)
                    )
                    words = (vocab_size + 31) // 32
                    bitmask = torch.zeros(
                        (2, words), device="cuda", dtype=torch.int32
                    )
                    bitmask[0, 0] = -2147483648
                    bitmask[1].fill_(-1)
                    if words > 1:
                        bitmask[0, 1] = 5
                    logits_before = logits.clone()
                    bitmask_before = bitmask.clone()

                    actual = MODULE.apply_token_bitmask(logits, bitmask)
                    expected = reference(logits, bitmask)

                    torch.testing.assert_close(
                        actual, expected, atol=0.0, rtol=0.0
                    )
                    torch.testing.assert_close(
                        logits, logits_before, atol=0.0, rtol=0.0
                    )
                    torch.testing.assert_close(
                        bitmask, bitmask_before, atol=0.0, rtol=0.0
                    )

    def test_out_of_place_strided_tail_and_sign_bit(self):
        logits_base = torch.arange(
            2 * 70, device="cuda", dtype=torch.float32
        ).reshape(2, 70)
        logits = logits_base[:, ::2]
        bitmask_base = torch.zeros((2, 4), device="cuda", dtype=torch.int32)
        bitmask_base[:, ::2] = torch.tensor(
            [
                [-2147483647, 4],
                [-1, 3],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        bitmask = bitmask_base[:, ::2]
        logits_before = logits.clone()
        bitmask_before = bitmask.clone()

        actual = MODULE.apply_token_bitmask(logits, bitmask)
        expected = reference(logits, bitmask)

        self.assertEqual(
            (actual.shape, actual.dtype), (logits.shape, logits.dtype)
        )
        self.assertNotEqual(actual.data_ptr(), logits.data_ptr())
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(logits, logits_before, atol=0.0, rtol=0.0)
        torch.testing.assert_close(bitmask, bitmask_before, atol=0.0, rtol=0.0)

    def test_supported_dtypes(self):
        bitmask = torch.tensor(
            [[-2147483648, 1]], device="cuda", dtype=torch.int32
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                logits = torch.linspace(-3, 3, 33, device="cuda").to(dtype)
                logits = logits.unsqueeze(0)

                actual = MODULE.apply_token_bitmask(logits, bitmask)

                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, reference(logits, bitmask), atol=0.0, rtol=0.0
                )

    def test_multiple_blocks_and_rows(self):
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        logits = torch.randn(
            (3, 513), generator=generator, device="cuda", dtype=torch.float32
        )
        bitmask = torch.randint(
            0,
            2**32,
            (3, 17),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        ).to(torch.int32)

        actual = MODULE.apply_token_bitmask(logits, bitmask)

        torch.testing.assert_close(
            actual, reference(logits, bitmask), atol=0.0, rtol=0.0
        )

    def test_empty_shapes(self):
        for shape in ((0, 35), (2, 0)):
            with self.subTest(shape=shape):
                logits = torch.empty(shape, device="cuda", dtype=torch.float16)
                bitmask = torch.empty(
                    (shape[0], (shape[1] + 31) // 32),
                    device="cuda",
                    dtype=torch.int32,
                )

                actual = MODULE.apply_token_bitmask(logits, bitmask)

                self.assertEqual(
                    (actual.shape, actual.dtype), (shape, logits.dtype)
                )
                self.assertIsNot(actual, logits)

    def test_vendor_grid_stride_above_grid_limit(self):
        batch_size, vocab_size = 256, 65537
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        logits = torch.randn(
            (batch_size, vocab_size),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        bitmask = torch.randint(
            0,
            2**32,
            (batch_size, (vocab_size + 31) // 32),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        ).to(torch.int32)
        expected = reference(logits, bitmask)

        for vendor, module in VENDOR_MODULES.items():
            with self.subTest(vendor=vendor):
                actual = module.apply_token_bitmask(logits, bitmask)
                torch.testing.assert_close(
                    actual, expected, atol=0.0, rtol=0.0
                )

    def test_enflame_large_block_all_dtypes(self):
        vocab_size = 12 * 4096 + 17
        generator = torch.Generator(device="cuda").manual_seed(20260824)
        bitmask = torch.randint(
            0,
            2**32,
            (1, (vocab_size + 31) // 32),
            generator=generator,
            device="cuda",
            dtype=torch.int64,
        ).to(torch.int32)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                logits = torch.randn(
                    (1, vocab_size),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                actual = VENDOR_MODULES["enflame"].apply_token_bitmask(
                    logits, bitmask
                )
                torch.testing.assert_close(
                    actual, reference(logits, bitmask), atol=0.0, rtol=0.0
                )


if __name__ == "__main__":
    unittest.main()

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
    / "apply_token_bitmask.py"
)
SPEC = importlib.util.spec_from_file_location(
    "apply_token_bitmask_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(logits, bitmask):
    vocab_size = logits.shape[1]
    token = torch.arange(vocab_size, device=logits.device)
    word = token // 32
    bit = token % 32
    allowed = ((bitmask[:, word] >> bit) & 1) != 0
    return torch.where(allowed, logits, torch.full_like(logits, -float("inf")))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ApplyTokenBitmaskTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

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
    / "silu_and_mul_masked.py"
)
SPEC = importlib.util.spec_from_file_location(
    "silu_and_mul_masked_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

KUNLUN_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_kunlunxin"
    / "ops"
    / "silu_and_mul_masked.py"
)
KUNLUN_SPEC = importlib.util.spec_from_file_location(
    "silu_and_mul_masked_kunlunxin_module", KUNLUN_MODULE_PATH
)
if KUNLUN_SPEC is None or KUNLUN_SPEC.loader is None:
    raise RuntimeError(f"cannot load {KUNLUN_MODULE_PATH}")
KUNLUN_MODULE = importlib.util.module_from_spec(KUNLUN_SPEC)
KUNLUN_SPEC.loader.exec_module(KUNLUN_MODULE)

AMD_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_amd"
    / "ops"
    / "silu_and_mul_masked.py"
)
AMD_SPEC = importlib.util.spec_from_file_location(
    "silu_and_mul_masked_amd_module", AMD_MODULE_PATH
)
if AMD_SPEC is None or AMD_SPEC.loader is None:
    raise RuntimeError(f"cannot load {AMD_MODULE_PATH}")
AMD_MODULE = importlib.util.module_from_spec(AMD_SPEC)
AMD_SPEC.loader.exec_module(AMD_MODULE)

ENFLAME_MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "runtime"
    / "backend"
    / "_enflame"
    / "ops"
    / "silu_and_mul_masked.py"
)
ENFLAME_SPEC = importlib.util.spec_from_file_location(
    "silu_and_mul_masked_enflame_module", ENFLAME_MODULE_PATH
)
if ENFLAME_SPEC is None or ENFLAME_SPEC.loader is None:
    raise RuntimeError(f"cannot load {ENFLAME_MODULE_PATH}")
ENFLAME_MODULE = importlib.util.module_from_spec(ENFLAME_SPEC)
ENFLAME_SPEC.loader.exec_module(ENFLAME_MODULE)


def reference(input, masked_m):
    experts, tokens, width = input.shape
    half_width = width // 2
    output = torch.zeros(
        (experts, tokens, half_width),
        dtype=torch.bfloat16,
        device=input.device,
    )
    for expert in range(experts):
        valid_rows = int(masked_m[expert].item())
        if valid_rows <= 0:
            continue
        gate = input[expert, :valid_rows, :half_width].float()
        up = input[expert, :valid_rows, half_width:].float()
        output[expert, :valid_rows] = (gate * torch.sigmoid(gate) * up).to(
            torch.bfloat16
        )
    return output


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class SiluAndMulMaskedTest(unittest.TestCase):
    def _check(self, input, masked_m, module=MODULE):
        snapshot = input.clone()
        mask_snapshot = masked_m.clone()
        expected = reference(input, masked_m)
        rows = torch.arange(input.shape[1], device="cuda")
        valid = rows[None, :] < masked_m[:, None]
        actual = module.silu_and_mul_masked(input, masked_m)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual[valid],
            expected[valid],
            atol=1.5e-2,
            rtol=1.5e-2,
            equal_nan=True,
        )
        torch.testing.assert_close(
            input, snapshot, atol=0, rtol=0, equal_nan=True
        )
        torch.testing.assert_close(masked_m, mask_snapshot, atol=0, rtol=0)

    def test_masks_and_column_tails(self):
        for module in (MODULE, KUNLUN_MODULE, AMD_MODULE, ENFLAME_MODULE):
            for integer_dtype in (torch.int32, torch.int64):
                for tokens, width in ((3, 16), (64, 256), (7, 2050)):
                    with self.subTest(
                        module=module.__name__,
                        integer_dtype=integer_dtype,
                        tokens=tokens,
                        width=width,
                    ):
                        input = torch.randn(
                            4,
                            tokens,
                            width,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        masked_m = torch.tensor(
                            [0, 1, tokens - 1, tokens],
                            device="cuda",
                            dtype=integer_dtype,
                        )
                        self._check(input, masked_m, module)

    def test_non_contiguous_input_and_mask(self):
        base = torch.randn(6, 10, 516, device="cuda", dtype=torch.bfloat16)
        input = base[::2, ::2, ::2]
        mask_base = torch.tensor(
            [5, -1, 3, -1, 0, -1], device="cuda", dtype=torch.int32
        )
        masked_m = mask_base[::2]
        self.assertFalse(input.is_contiguous())
        self.assertFalse(masked_m.is_contiguous())
        for module in (MODULE, KUNLUN_MODULE, AMD_MODULE, ENFLAME_MODULE):
            with self.subTest(module=module.__name__):
                self._check(input, masked_m, module)

    def test_grid_stride_fold_path(self):
        experts, tokens = 512, 257
        input = torch.randn(
            experts, tokens, 2, device="cuda", dtype=torch.bfloat16
        )
        masked_m = torch.tensor(
            [0, 1, tokens - 1, tokens] * (experts // 4),
            device="cuda",
            dtype=torch.int32,
        )
        self.assertGreater(experts * tokens, 2 * 65535)
        self.assertEqual(255 * tokens, 65535)
        self.assertEqual(
            (masked_m[0].item(), masked_m[255].item()),
            (0, tokens),
        )
        for module in (MODULE, KUNLUN_MODULE, AMD_MODULE, ENFLAME_MODULE):
            with self.subTest(module=module.__name__):
                self._check(input, masked_m, module)

    def test_special_values(self):
        largest = torch.finfo(torch.bfloat16).max
        gate = torch.tensor(
            [
                float("-inf"),
                -100.0,
                -92.0,
                -90.0,
                -88.0,
                -20.0,
                -0.0,
                0.0,
                20.0,
                100.0,
                float("inf"),
                float("nan"),
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        up = torch.tensor(
            [
                0.0,
                2.0,
                largest,
                largest,
                largest,
                -3.0,
                4.0,
                -4.0,
                3.0,
                -2.0,
                0.0,
                1.0,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        input = torch.cat((gate, up)).reshape(1, 1, -1)
        masked_m = torch.ones(1, device="cuda", dtype=torch.int32)
        for module in (MODULE, KUNLUN_MODULE, AMD_MODULE, ENFLAME_MODULE):
            with self.subTest(module=module.__name__):
                self._check(input, masked_m, module)

    def test_empty_dimensions(self):
        self._check(
            torch.empty(0, 3, 8, device="cuda", dtype=torch.bfloat16),
            torch.empty(0, device="cuda", dtype=torch.int32),
        )
        self._check(
            torch.empty(2, 0, 8, device="cuda", dtype=torch.bfloat16),
            torch.zeros(2, device="cuda", dtype=torch.int32),
        )
        self._check(
            torch.empty(2, 3, 0, device="cuda", dtype=torch.bfloat16),
            torch.full((2,), 3, device="cuda", dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()

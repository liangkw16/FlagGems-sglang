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
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "sgemm_lora_b.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sgemm_lora_b_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_batch_info(
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation=None,
):
    seg_indptr = torch.tensor(seg_indptr, device="cuda", dtype=torch.int32)
    seg_lens = seg_indptr[1:] - seg_indptr[:-1]
    return SimpleNamespace(
        bs=len(weight_indices),
        max_len=int(seg_lens.max().item()) if len(weight_indices) else 0,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        weight_indices=torch.tensor(
            weight_indices, device="cuda", dtype=torch.int32
        ),
        lora_ranks=torch.tensor(lora_ranks, device="cuda", dtype=torch.int32),
        scalings=torch.tensor(scalings, device="cuda", dtype=torch.float32),
        permutation=(
            torch.tensor(permutation, device="cuda", dtype=torch.int64)
            if permutation is not None
            else None
        ),
    )


def reference(x, weights, batch_info, base_output):
    output = base_output.clone().float()
    for batch_id in range(batch_info.bs):
        start = int(batch_info.seg_indptr[batch_id].item())
        end = int(batch_info.seg_indptr[batch_id + 1].item())
        if start == end:
            continue
        weight_index = int(batch_info.weight_indices[batch_id].item())
        if int(batch_info.lora_ranks[weight_index].item()) == 0:
            continue
        scaling = float(batch_info.scalings[weight_index].item())
        if batch_info.permutation is None:
            rows = torch.arange(start, end, device=x.device)
        else:
            rows = batch_info.permutation[start:end].long()
        output[rows] += scaling * (
            x[rows].float() @ weights[weight_index].float().t()
        )
    return output.to(base_output.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class SgemmLoraBTest(unittest.TestCase):
    def test_segments_rank_zero_and_dtypes_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        info = make_batch_info(
            [0, 2, 2, 5, 7], [0, 1, 1, 2], [5, 0, 3], [0.5, 1.0, -0.25]
        )
        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                x = torch.arange(7 * 5, device="cuda").reshape(7, 5).to(dtype)
                weights = (
                    torch.arange(3 * 9 * 5, device="cuda")
                    .reshape(3, 9, 5)
                    .to(dtype)
                    / 100
                )
                base = (
                    torch.linspace(-1, 1, 7 * 9, device="cuda")
                    .reshape(7, 9)
                    .to(dtype)
                )
                before = (x.clone(), weights.clone(), base.clone())

                actual = MODULE.sgemm_lora_b(x, weights, info, base)
                expected = reference(x, weights, info, base)

                self.assertEqual(actual.shape, base.shape)
                self.assertEqual(actual.dtype, base.dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                for value, original in zip((x, weights, base), before):
                    torch.testing.assert_close(
                        value, original, atol=0.0, rtol=0.0
                    )

    def test_permutation_real_strides_and_full_k(self):
        info = make_batch_info(
            [0, 3, 6],
            [0, 1],
            [2, 1],
            [1.25, -0.5],
            permutation=[4, 0, 5, 2, 1, 3],
        )
        x = torch.arange(6 * 14, device="cuda", dtype=torch.float32).reshape(
            6, 14
        )[:, 1::2]
        weights = torch.arange(
            2 * 22 * 14, device="cuda", dtype=torch.float32
        ).reshape(2, 22, 14)[:, 1::2, ::2]
        base = torch.arange(
            6 * 22, device="cuda", dtype=torch.float32
        ).reshape(6, 22)[:, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertFalse(weights.is_contiguous())
        self.assertFalse(base.is_contiguous())

        actual = MODULE.sgemm_lora_b(x, weights, info, base)
        expected = reference(x, weights, info, base)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_empty_input(self):
        info = make_batch_info([], [], [], [])
        x = torch.empty((0, 7), device="cuda")
        weights = torch.empty((0, 11, 7), device="cuda")
        base = torch.empty((0, 11), device="cuda")

        actual = MODULE.sgemm_lora_b(x, weights, info, base)

        self.assertEqual(actual.shape, base.shape)
        self.assertEqual(actual.dtype, base.dtype)


if __name__ == "__main__":
    unittest.main()

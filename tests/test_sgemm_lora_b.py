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
    "sgemm_lora_b_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "sgemm_lora_b.py",
)
ASCEND_MODULE = _load_module(
    "sgemm_lora_b_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "sgemm_lora_b.py",
)
ILUVATAR_MODULE = _load_module(
    "sgemm_lora_b_iluvatar_module",
    BACKEND_ROOT / "_iluvatar" / "ops" / "sgemm_lora_b.py",
)
ENFLAME_MODULE = _load_module(
    "sgemm_lora_b_enflame_module",
    BACKEND_ROOT / "_enflame" / "ops" / "sgemm_lora_b.py",
)
KUNLUN_MODULE = _load_module(
    "sgemm_lora_b_kunlunxin_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "sgemm_lora_b.py",
)


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

    def test_seg_indptr_is_authoritative_and_empty_metadata_is_ignored(self):
        info = SimpleNamespace(
            bs=3,
            max_len=17,
            seg_lens=torch.tensor([1, 0, 1], device="cuda"),
            seg_indptr=torch.tensor(
                [0, -1, 17, -1, 17, -1, 18, -1], device="cuda"
            )[::2],
            weight_indices=torch.tensor(
                [0, 1 << 28, 0], device="cuda", dtype=torch.int64
            ),
            lora_ranks=torch.tensor([65], device="cuda"),
            scalings=torch.tensor([0.5], device="cuda"),
            permutation=None,
        )
        x = (
            torch.arange(18 * 65, device="cuda", dtype=torch.float32)
            .reshape(18, 65)
            .div(100)
        )
        weights = (
            torch.arange(67 * 65, device="cuda", dtype=torch.float32)
            .reshape(1, 67, 65)
            .div(1000)
        )
        base = torch.linspace(
            -1, 1, 18 * 67, device="cuda", dtype=torch.float32
        ).reshape(18, 67)

        actual = MODULE.sgemm_lora_b(x, weights, info, base)
        expected = reference(x, weights, info, base)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(info.seg_indptr.stride(), (2,))

    def test_empty_input(self):
        info = make_batch_info([], [], [], [])
        x = torch.empty((0, 7), device="cuda")
        weights = torch.empty((0, 11, 7), device="cuda")
        base = torch.empty((0, 11), device="cuda")

        actual = MODULE.sgemm_lora_b(x, weights, info, base)

        self.assertEqual(actual.shape, base.shape)
        self.assertEqual(actual.dtype, base.dtype)

    def test_kunlun_regular_bmm_handles_ragged_strided_inputs(self):
        torch.manual_seed(20260826)
        seg_indptr = torch.tensor(
            [0, -1, 17, -1, 17, -1, 18, -1, 51, -1], device="cuda"
        )[::2]
        weight_indices = torch.tensor(
            [0, -1, 1 << 28, -1, 1, -1, 2, -1], device="cuda"
        )[::2]
        lora_ranks = torch.tensor([1, -1, 0, -1, 65, -1], device="cuda")[::2]
        scalings = torch.tensor(
            [0.75, 0.0, 1.0, 0.0, -0.5, 0.0], device="cuda"
        )[::2]
        permutation_storage = torch.empty(
            51 * 2, device="cuda", dtype=torch.int64
        )
        permutation_storage[::2] = torch.randperm(51, device="cuda")
        permutation = permutation_storage[::2]
        info = SimpleNamespace(
            bs=4,
            max_len=33,
            seg_lens=seg_indptr[1:] - seg_indptr[:-1],
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
        )
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                x = (torch.randn((51, 130), device="cuda", dtype=dtype) * 0.1)[
                    :, ::2
                ]
                weights = (
                    torch.randn((3, 134, 130), device="cuda", dtype=dtype)
                    * 0.1
                )[:, ::2, ::2]
                base = (
                    torch.randn((51, 134), device="cuda", dtype=dtype) * 0.01
                )[:, ::2]
                for value in (x, weights, base):
                    self.assertFalse(value.is_contiguous())
                values = (
                    x,
                    weights,
                    base,
                    seg_indptr,
                    weight_indices,
                    lora_ranks,
                    scalings,
                    permutation,
                )
                before = tuple(value.clone() for value in values)

                actual = KUNLUN_MODULE.sgemm_lora_b(x, weights, info, base)
                expected = reference(x, weights, info, base)
                torch.cuda.synchronize()

                self.assertEqual(actual.shape, base.shape)
                self.assertEqual(actual.dtype, base.dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                for value, original in zip(values, before):
                    torch.testing.assert_close(
                        value, original, atol=0.0, rtol=0.0
                    )

    def test_vendors_cover_fold_and_split_fp16(self):
        torch.manual_seed(20260824)
        bs, max_len, rank, out_dim = 8, 2048, 64, 4096
        lens = torch.randint(700, max_len + 1, (bs,)).tolist()
        seg = [0]
        for length in lens:
            seg.append(seg[-1] + length)
        total = seg[-1]
        tiles = ((max_len + 63) // 64) * ((out_dim + 127) // 128)
        self.assertGreater(tiles * bs, 4096)
        self.assertEqual(
            bs * ((max_len + 31) // 32) * ((out_dim + 31) // 32),
            65536,
        )
        self.assertEqual(
            bs * ((max_len + 31) // 32) * ((out_dim + 63) // 64),
            32768,
        )
        perm = torch.randperm(total).tolist()

        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
            (torch.bfloat16, 1.5e-2),
        ):
            with self.subTest(dtype=dtype):
                x = torch.randn((total, rank), device="cuda", dtype=dtype)
                weights = torch.randn(
                    (bs, out_dim, rank), device="cuda", dtype=dtype
                )
                base = (
                    torch.randn((total, out_dim), device="cuda", dtype=dtype)
                    * 0.01
                )
                info = make_batch_info(
                    seg, list(range(bs)), [rank] * bs, [1.0] * bs, perm
                )
                expected = reference(x, weights, info, base)

                for name, module in (
                    ("generic", MODULE),
                    ("ascend", ASCEND_MODULE),
                    ("iluvatar", ILUVATAR_MODULE),
                    ("enflame", ENFLAME_MODULE),
                    ("kunlunxin", KUNLUN_MODULE),
                ):
                    with self.subTest(module=name):
                        actual = module.sgemm_lora_b(x, weights, info, base)
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=tolerance,
                            rtol=tolerance,
                        )


if __name__ == "__main__":
    unittest.main()

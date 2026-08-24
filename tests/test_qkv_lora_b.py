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
    "qkv_lora_b_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "qkv_lora_b.py",
)
ASCEND_MODULE = _load_module(
    "qkv_lora_b_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "qkv_lora_b.py",
)
ILUVATAR_MODULE = _load_module(
    "qkv_lora_b_iluvatar_module",
    BACKEND_ROOT / "_iluvatar" / "ops" / "qkv_lora_b.py",
)
ENFLAME_MODULE = _load_module(
    "qkv_lora_b_enflame_module",
    BACKEND_ROOT / "_enflame" / "ops" / "qkv_lora_b.py",
)
KUNLUN_MODULE = _load_module(
    "qkv_lora_b_kunlunxin_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "qkv_lora_b.py",
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


def reference(
    x,
    qkv_lora_b,
    batch_info,
    output_offset,
    max_qkv_out_dim,
    base_output,
):
    output = base_output.clone().float()
    n_slices = output_offset.numel() - 1
    rank = qkv_lora_b.shape[-1]
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
        x_segment = x[rows].float()
        for slice_id in range(n_slices):
            output_start = int(output_offset[slice_id].item())
            output_end = int(output_offset[slice_id + 1].item())
            slice_start = slice_id * rank
            slice_end = (slice_id + 1) * rank
            x_slice = x_segment[:, slice(slice_start, slice_end)]
            weights = qkv_lora_b[
                weight_index, output_start:output_end, :
            ].float()
            output[rows, output_start:output_end] += scaling * (
                x_slice @ weights.t()
            )
    return output.to(base_output.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class QkvLoraBTest(unittest.TestCase):
    def test_dtypes_slices_empty_segment_and_rank_zero(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        info = make_batch_info(
            [0, 2, 2, 5, 7],
            [0, 1, 1, 2],
            [5, 0, 5],
            [0.5, 1.0, -0.25],
        )
        output_offset = torch.tensor(
            [0, 5, 8, 15], device="cuda", dtype=torch.int32
        )

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                x = (
                    torch.arange(7 * 15, device="cuda")
                    .reshape(7, 15)
                    .to(dtype)
                    / 100
                )
                weights = (
                    torch.arange(3 * 15 * 5, device="cuda")
                    .reshape(3, 15, 5)
                    .to(dtype)
                    / 200
                )
                base = (
                    torch.linspace(-1, 1, 7 * 15, device="cuda")
                    .reshape(7, 15)
                    .to(dtype)
                )
                before = (x.clone(), weights.clone(), base.clone())

                actual = MODULE.qkv_lora_b(
                    x, weights, info, output_offset, 7, base
                )
                expected = reference(x, weights, info, output_offset, 7, base)

                self.assertEqual(actual.shape, base.shape)
                self.assertEqual(actual.dtype, base.dtype)
                self.assertNotEqual(actual.data_ptr(), base.data_ptr())
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                for value, original in zip((x, weights, base), before):
                    torch.testing.assert_close(value, original, atol=0, rtol=0)

    def test_permutation_noncontiguous_and_dynamic_slice_count(self):
        info = make_batch_info(
            [0, 3, 6],
            [0, 1],
            [7, 7],
            [1.25, -0.5],
            permutation=[4, 0, 5, 2, 1, 3],
        )
        x = torch.arange(6 * 28, device="cuda", dtype=torch.float32).reshape(
            6, 28
        )[:, ::2]
        weights = torch.arange(
            2 * 22 * 14, device="cuda", dtype=torch.float32
        ).reshape(2, 22, 14)[:, ::2, ::2]
        base = torch.arange(
            6 * 22, device="cuda", dtype=torch.float32
        ).reshape(6, 22)[:, ::2]
        output_offset = torch.tensor(
            [0, 99, 4, 99, 11], device="cuda", dtype=torch.int32
        )[::2]
        self.assertEqual(output_offset.tolist(), [0, 4, 11])
        for value in (x, weights, base, output_offset):
            self.assertFalse(value.is_contiguous())
        before = (
            x.clone(),
            weights.clone(),
            base.clone(),
            output_offset.clone(),
            info.permutation.clone(),
        )

        actual = MODULE.qkv_lora_b(x, weights, info, output_offset, 7, base)
        expected = reference(x, weights, info, output_offset, 7, base)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        for value, original in zip(
            (x, weights, base, output_offset, info.permutation), before
        ):
            torch.testing.assert_close(value, original, atol=0, rtol=0)

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
            lora_ranks=torch.tensor([33], device="cuda"),
            scalings=torch.tensor([0.5], device="cuda"),
            permutation=None,
        )
        output_offset = torch.tensor([0, -1, 65, -1, 66, -1], device="cuda")[
            ::2
        ]
        x = (
            torch.arange(18 * 66, device="cuda", dtype=torch.float32)
            .reshape(18, 66)
            .div(100)
        )
        weights = (
            torch.arange(66 * 33, device="cuda", dtype=torch.float32)
            .reshape(1, 66, 33)
            .div(1000)
        )
        base = torch.linspace(
            -1, 1, 18 * 66, device="cuda", dtype=torch.float32
        ).reshape(18, 66)

        actual = MODULE.qkv_lora_b(x, weights, info, output_offset, 65, base)
        expected = reference(x, weights, info, output_offset, 65, base)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(info.seg_indptr.stride(), (2,))
        self.assertEqual(output_offset.stride(), (2,))

    def test_vendors_cover_fold_and_split_fp16(self):
        torch.manual_seed(20260824)
        bs, max_len, rank = 4, 4096, 64
        slice_sizes = [768, 512, 1024]
        out_width = sum(slice_sizes)
        max_qkv_out_dim = max(slice_sizes)
        lens = torch.randint(3000, max_len + 1, (bs,)).tolist()
        seg = [0]
        for length in lens:
            seg.append(seg[-1] + length)
        total = seg[-1]
        n_slices = len(slice_sizes)
        tiles = ((max_len + 63) // 64) * ((max_qkv_out_dim + 127) // 128)
        self.assertGreater(tiles * n_slices * bs, 4096)
        perm = torch.randperm(total).tolist()
        offsets = [0]
        for size in slice_sizes:
            offsets.append(offsets[-1] + size)
        output_offset = torch.tensor(offsets, device="cuda")

        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
            (torch.bfloat16, 1.5e-2),
        ):
            with self.subTest(dtype=dtype):
                x = torch.randn(
                    (total, n_slices * rank), device="cuda", dtype=dtype
                )
                weights = torch.randn(
                    (bs, out_width, rank), device="cuda", dtype=dtype
                )
                base = (
                    torch.randn((total, out_width), device="cuda", dtype=dtype)
                    * 0.01
                )
                info = make_batch_info(
                    seg, list(range(bs)), [rank] * bs, [1.0] * bs, perm
                )
                expected = reference(
                    x, weights, info, output_offset, max_qkv_out_dim, base
                )

                for name, module in (
                    ("generic", MODULE),
                    ("ascend", ASCEND_MODULE),
                    ("iluvatar", ILUVATAR_MODULE),
                    ("enflame", ENFLAME_MODULE),
                    ("kunlunxin", KUNLUN_MODULE),
                ):
                    with self.subTest(module=name):
                        actual = module.qkv_lora_b(
                            x,
                            weights,
                            info,
                            output_offset,
                            max_qkv_out_dim,
                            base,
                        )
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=tolerance,
                            rtol=tolerance,
                        )


if __name__ == "__main__":
    unittest.main()

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

import ast
import importlib.util
import os
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
KUNLUN_SOURCE = BACKEND_ROOT / "_kunlunxin" / "ops" / "sgemm_lora_b.py"


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


class SgemmLoraBKunlunRoutingTest(unittest.TestCase):
    def test_layout_copy_and_safe_adapter_are_routed_to_regular_bmm(self):
        seg_indptr = torch.tensor([0, -1, 2, -1, 2, -1])[::2]
        weight_indices = torch.tensor([1, -1, 1 << 28, -1])[::2]
        lora_ranks = torch.tensor([3, 3])
        scalings = torch.tensor([0.5, -0.25])
        permutation = torch.tensor([1, -1, 0, -1])[::2]
        info = SimpleNamespace(
            bs=2,
            max_len=2,
            seg_lens=seg_indptr[1:] - seg_indptr[:-1],
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            permutation=permutation,
        )
        x = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)[:, ::2]
        weights = torch.arange(2 * 140 * 6, dtype=torch.float32).reshape(
            2, 140, 6
        )[:, ::2, ::2]
        base = torch.arange(2 * 140, dtype=torch.float32).reshape(2, 140)[
            :, ::2
        ]
        before = tuple(
            value.clone()
            for value in (
                x,
                weights,
                base,
                seg_indptr,
                weight_indices,
                lora_ranks,
                scalings,
                permutation,
            )
        )
        simulation_flags = (
            "TRITONXPU_OTHER_SIM",
            "TRITONXPU_STORE_MASK_SIM",
        )

        with (
            patch.dict(os.environ, dict.fromkeys(simulation_flags, "0")),
            patch.object(KUNLUN_MODULE, "_safe_adapter_kernel") as safe_kernel,
            patch.object(KUNLUN_MODULE, "_pack_x_kernel") as pack_x_kernel,
            patch.object(KUNLUN_MODULE, "_regular_bmm_kernel") as bmm_kernel,
            patch.object(
                KUNLUN_MODULE, "_scatter_add_kernel"
            ) as scatter_kernel,
            patch.object(KUNLUN_MODULE, "_MAX_GRID_SIZE", 1),
        ):
            launch_order = []

            def record_launch(name, *args, **kwargs):
                launch_order.append(
                    (name,)
                    + tuple(os.environ.get(key) for key in simulation_flags)
                )

            for kernel, name in (
                (safe_kernel, "safe"),
                (pack_x_kernel, "pack"),
                (bmm_kernel, "bmm"),
                (scatter_kernel, "scatter"),
            ):
                kernel.__getitem__.return_value.side_effect = partial(
                    record_launch, name
                )
            actual = KUNLUN_MODULE.sgemm_lora_b(x, weights, info, base)
            self.assertTrue(
                all(key not in os.environ for key in simulation_flags)
            )

        safe_args = safe_kernel.__getitem__.return_value.call_args.args
        bmm_args = bmm_kernel.__getitem__.return_value.call_args.args
        transposed_weights = bmm_args[1]
        safe_adapters = bmm_args[2]
        expected_weights = weights.transpose(1, 2).contiguous()

        self.assertIs(safe_args[0], seg_indptr)
        self.assertIs(safe_args[1], weight_indices)
        self.assertIs(safe_args[2], lora_ranks)
        self.assertEqual(safe_args[3].data_ptr(), safe_adapters.data_ptr())
        self.assertEqual(safe_adapters.dtype, torch.int32)
        self.assertEqual(safe_adapters.shape, (info.bs,))
        self.assertTrue(transposed_weights.is_contiguous())
        self.assertEqual(transposed_weights.shape, (2, 3, 70))
        self.assertEqual(transposed_weights.stride(), (210, 70, 1))
        torch.testing.assert_close(transposed_weights, expected_weights)
        torch.testing.assert_close(actual, base)
        self.assertTrue(pack_x_kernel.__getitem__.return_value.called)
        self.assertTrue(bmm_kernel.__getitem__.return_value.called)
        self.assertTrue(scatter_kernel.__getitem__.return_value.called)
        for kernel in (
            safe_kernel,
            pack_x_kernel,
            bmm_kernel,
            scatter_kernel,
        ):
            for call in kernel.__getitem__.return_value.call_args_list:
                self.assertNotIn("is_use_mask_zero", call.kwargs)
        scatter_calls = scatter_kernel.__getitem__.return_value.call_args_list
        self.assertEqual(
            launch_order,
            [("safe", "1", "1")]
            + [("pack", "1", "1")] * 2
            + [("bmm", None, None)] * 6
            + [("scatter", "1", "1")] * 4,
        )
        self.assertEqual(
            [call.args[9] for call in scatter_calls], [0, 1, 2, 3]
        )
        self.assertEqual(
            [call.kwargs["BLOCK_N"] for call in scatter_calls], [256] * 4
        )
        for value, original in zip(
            (
                x,
                weights,
                base,
                seg_indptr,
                weight_indices,
                lora_ranks,
                scalings,
                permutation,
            ),
            before,
        ):
            torch.testing.assert_close(value, original, atol=0.0, rtol=0.0)

    def test_wrapper_uses_only_one_layout_copy_and_no_torch_core_compute(self):
        source = KUNLUN_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "sgemm_lora_b"
        )

        layout_copies = []
        forbidden_methods = {
            "bmm",
            "einsum",
            "gather",
            "index_select",
            "item",
            "matmul",
            "mm",
            "tolist",
        }
        torch_calls = set()
        forbidden_calls = set()
        for node in ast.walk(wrapper):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                if node.func.attr in forbidden_methods:
                    forbidden_calls.add(node.func.attr)
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    torch_calls.add(node.func.attr)
                if (
                    node.func.attr == "contiguous"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Attribute)
                    and node.func.value.func.attr == "transpose"
                ):
                    layout_copies.append(node.func.value)

        self.assertEqual(forbidden_calls, set())
        self.assertEqual(torch_calls, {"empty"})
        self.assertFalse(
            any(isinstance(node, ast.MatMult) for node in ast.walk(wrapper))
        )
        self.assertFalse(
            any(isinstance(node, ast.Try) for node in ast.walk(wrapper))
        )
        self.assertEqual(len(layout_copies), 1)
        self.assertEqual(
            [argument.value for argument in layout_copies[0].args], [1, 2]
        )
        self.assertIn("def _safe_adapter_kernel", source)
        self.assertNotIn("def _pack_weights_kernel", source)

        scatter = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_scatter_add_kernel"
        )
        scatter_source = ast.get_source_segment(source, scatter)
        self.assertIsNotNone(scatter_source)
        self.assertFalse(
            any(isinstance(node, ast.Return) for node in ast.walk(scatter))
        )
        self.assertIn(
            "BLOCK_N", [argument.arg for argument in scatter.args.args]
        )
        self.assertNotIn("BLOCK_M", scatter_source)
        self.assertNotIn("BLOCK_SIZE", scatter_source)
        self.assertNotIn("logical_offsets", scatter_source)
        self.assertNotIn("total_elements", scatter_source)
        self.assertEqual(scatter_source.count("tl.arange"), 1)
        self.assertIn(
            "logical_pid = program_start + tl.program_id(0)", scatter_source
        )
        self.assertIn("row_pid = logical_pid // nblocks", scatter_source)
        self.assertIn(
            "cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)",
            scatter_source,
        )
        self.assertLess(
            scatter_source.index("token_active ="),
            scatter_source.index("weight_index ="),
        )
        self.assertLess(
            scatter_source.index("weight_index ="),
            scatter_source.index("lora_rank ="),
        )
        self.assertGreaterEqual(scatter_source.count("mask=token_active"), 2)
        self.assertGreaterEqual(scatter_source.count("mask=active"), 2)
        self.assertGreaterEqual(scatter_source.count("mask=value_mask"), 3)

        loops = [node for node in wrapper.body if isinstance(node, ast.For)]
        loop_sources = [ast.get_source_segment(source, node) for node in loops]
        bmm_loop = next(
            node
            for node, loop_source in zip(loops, loop_sources)
            if "_regular_bmm_kernel" in loop_source
        )
        scatter_loop = next(
            node
            for node, loop_source in zip(loops, loop_sources)
            if "_scatter_add_kernel" in loop_source
        )
        self.assertLess(bmm_loop.lineno, scatter_loop.lineno)
        self.assertNotIn(
            "_scatter_add_kernel", ast.get_source_segment(source, bmm_loop)
        )


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

    def test_kunlun_rank_zero_adapter_does_not_read_missing_weight(self):
        info = make_batch_info([0, 2], [1], [3, 0], [1.0, 1.0])
        x = torch.randn((2, 3), device="cuda")
        weights = torch.randn((1, 5, 3), device="cuda")
        base = torch.randn((2, 5), device="cuda")

        actual = KUNLUN_MODULE.sgemm_lora_b(x, weights, info, base)

        torch.testing.assert_close(actual, base, atol=0.0, rtol=0.0)

    def test_kunlun_empty_segments_allow_zero_weights(self):
        info = SimpleNamespace(
            bs=1,
            max_len=1,
            seg_lens=torch.tensor([0], device="cuda"),
            seg_indptr=torch.tensor([0, 0], device="cuda"),
            weight_indices=torch.tensor([1 << 28], device="cuda"),
            lora_ranks=torch.empty(0, device="cuda", dtype=torch.int32),
            scalings=torch.empty(0, device="cuda"),
            permutation=None,
        )
        x = torch.empty((0, 3), device="cuda")
        weights = torch.empty((0, 5, 3), device="cuda")
        base = torch.randn((1, 5), device="cuda")

        actual = KUNLUN_MODULE.sgemm_lora_b(x, weights, info, base)

        torch.testing.assert_close(actual, base, atol=0.0, rtol=0.0)

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
        self.assertEqual((65536 + 65535 - 1) // 65535, 2)
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

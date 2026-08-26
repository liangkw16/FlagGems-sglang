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
import inspect
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

OPS_PATH = Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops"
BACKEND_PATH = OPS_PATH.parent / "runtime" / "backend"


def _load_module(name, path=None):
    if path is None:
        path = OPS_PATH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTEXT_ATTENTION = _load_module("context_attention")
CONTEXT_ATTENTION_METAX = _load_module(
    "context_attention_metax",
    BACKEND_PATH / "_metax" / "ops" / "context_attention.py",
)
CONTEXT_ATTENTION_ASCEND = _load_module(
    "context_attention_ascend",
    BACKEND_PATH / "_ascend" / "ops" / "context_attention.py",
)
CONTEXT_ATTENTION_KUNLUNXIN = _load_module(
    "context_attention_kunlunxin",
    BACKEND_PATH / "_kunlunxin" / "ops" / "context_attention.py",
)
CONTEXT_ATTENTION_MODULES = (
    CONTEXT_ATTENTION,
    CONTEXT_ATTENTION_METAX,
    CONTEXT_ATTENTION_ASCEND,
    CONTEXT_ATTENTION_KUNLUNXIN,
)


def reference(q, k, v, starts, lengths, is_causal):
    output = torch.empty_like(q, dtype=torch.float32)
    for start_tensor, length_tensor in zip(starts, lengths):
        start = int(start_tensor.item())
        length = int(length_tensor.item())
        end = start + length
        output[start:end] = F.scaled_dot_product_attention(
            q[start:end].permute(1, 0, 2).float(),
            k[start:end].permute(1, 0, 2).float(),
            v[start:end].permute(1, 0, 2).float(),
            is_causal=is_causal,
        ).permute(1, 0, 2)
    return output


def make_case(
    dtype,
    sequence_lengths=None,
    num_heads=3,
    head_dim=37,
):
    if sequence_lengths is None:
        sequence_lengths = [1, 5, 33, 7]
    total_tokens = sum(sequence_lengths)
    generator = torch.Generator(device="cuda").manual_seed(
        20260824 + total_tokens + head_dim
    )

    tensors = []
    for _ in range(3):
        base = torch.randn(
            (total_tokens, num_heads, head_dim * 2),
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        tensors.append(base[..., ::2])

    starts_values = torch.tensor(
        [
            sum(sequence_lengths[:index])
            for index in range(len(sequence_lengths))
        ],
        device="cuda",
        dtype=torch.int64,
    )
    lengths_values = torch.tensor(
        sequence_lengths, device="cuda", dtype=torch.int64
    )
    starts_base = torch.empty(
        starts_values.numel() * 2, device="cuda", dtype=torch.int64
    )
    lengths_base = torch.empty(
        lengths_values.numel() * 2, device="cuda", dtype=torch.int64
    )
    starts_base[::2] = starts_values
    lengths_base[::2] = lengths_values
    return (*tensors, starts_base[::2], lengths_base[::2])


class ContextAttentionHostTest(unittest.TestCase):
    def test_small_head_dim_and_vendor_launch_policies(self):
        modules = (
            (CONTEXT_ATTENTION, "_context_attention_kernel", 64, 4),
            (CONTEXT_ATTENTION_METAX, "_context_attention_kernel", 16, 4),
            (CONTEXT_ATTENTION_ASCEND, "_context_attention_kernel", 32, 4),
            (
                CONTEXT_ATTENTION_KUNLUNXIN,
                "_context_attention_static_q_kernel",
                32,
                4,
            ),
        )
        q = torch.empty((3, 1, 8), dtype=torch.float16)
        starts = torch.tensor([0], dtype=torch.int32)
        lengths = torch.tensor([3], dtype=torch.int32)

        for module, kernel_name, block_n, num_warps in modules:
            for is_causal in (False, True):
                launches = []

                class FakeKernel:
                    def __getitem__(self, grid):
                        def launch(*args, **kwargs):
                            launches.append((grid, kwargs))

                        return launch

                with mock.patch.object(module, kernel_name, FakeKernel()):
                    module.context_attention(
                        q, q, q, starts, lengths, 3, is_causal
                    )

                self.assertTrue(launches)
                for grid, kwargs in launches:
                    self.assertEqual(len(grid), 1)
                    self.assertEqual(kwargs["BLOCK_D"], 16)
                    self.assertEqual(kwargs["BLOCK_N"], block_n)
                    self.assertEqual(kwargs["num_warps"], num_warps)

    def test_ascend_and_kunlunxin_launch_caps(self):
        launches = []

        class FakeKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append(
                        (
                            grid[0],
                            kwargs["q_block_start"],
                            kwargs["q_programs"],
                        )
                    )

                return launch

        starts = torch.tensor([0], dtype=torch.int32)
        with mock.patch.object(
            CONTEXT_ATTENTION_ASCEND,
            "_context_attention_kernel",
            FakeKernel(),
        ):
            q = torch.empty((135168, 8, 1), dtype=torch.uint8)
            lengths = torch.tensor([135168], dtype=torch.int32)
            CONTEXT_ATTENTION_ASCEND.context_attention(
                q, q, q, starts, lengths, 135168, False
            )
        self.assertEqual(launches, [(29568, 0, 4224), (4224, 0, 4224)])
        self.assertLessEqual(max(grid for grid, _, _ in launches), 32768)

        launches.clear()
        with mock.patch.object(
            CONTEXT_ATTENTION_KUNLUNXIN,
            "_context_attention_static_q_kernel",
            FakeKernel(),
        ):
            q = torch.empty((65536 * 64, 1, 1), dtype=torch.uint8)
            lengths = torch.tensor([1], dtype=torch.int32)
            CONTEXT_ATTENTION_KUNLUNXIN.context_attention(
                q, q, q, starts, lengths, 1, False
            )
        self.assertEqual(launches, [(65535, 0, 65535), (1, 65535, 1)])
        self.assertLessEqual(max(grid for grid, _, _ in launches), 65535)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ContextAttentionTest(unittest.TestCase):
    def test_public_contract(self):
        for module in CONTEXT_ATTENTION_MODULES:
            function = module.context_attention
            self.assertEqual(
                list(inspect.signature(function).parameters),
                [
                    "q",
                    "k",
                    "v",
                    "b_start_loc",
                    "b_seq_len",
                    "max_input_len",
                    "is_causal",
                ],
            )
            self.assertEqual(module.__all__, ["context_attention"])

    def test_variable_lengths_causal_boundaries_strides_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for is_causal in (False, True):
                with self.subTest(dtype=dtype, is_causal=is_causal):
                    q, k, v, starts, lengths = make_case(dtype)
                    snapshots = tuple(
                        tensor.clone() for tensor in (q, k, v, starts, lengths)
                    )

                    expected = reference(q, k, v, starts, lengths, is_causal)
                    for module in CONTEXT_ATTENTION_MODULES:
                        actual = module.context_attention(
                            q,
                            k,
                            v,
                            starts,
                            lengths,
                            1,
                            is_causal,
                        )
                        self.assertEqual(
                            (actual.shape, actual.dtype),
                            (q.shape, torch.float32),
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=1e-2, rtol=1e-2
                        )
                    for tensor, snapshot in zip(
                        (q, k, v, starts, lengths), snapshots
                    ):
                        torch.testing.assert_close(
                            tensor, snapshot, atol=0.0, rtol=0.0
                        )

    def test_empty_packed_batch(self):
        q = torch.empty((0, 3, 37), device="cuda", dtype=torch.float16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        starts = torch.empty(0, device="cuda", dtype=torch.int32)
        lengths = torch.empty(0, device="cuda", dtype=torch.int32)

        for module in CONTEXT_ATTENTION_MODULES:
            actual = module.context_attention(
                q, k, v, starts, lengths, 0, True
            )
            self.assertEqual(
                (actual.shape, actual.dtype), (q.shape, torch.float32)
            )

    def test_head_dim_eight_and_underreported_max_input_len(self):
        q, k, v, starts, lengths = make_case(
            torch.float16,
            sequence_lengths=[3, 33],
            num_heads=2,
            head_dim=8,
        )
        for is_causal in (False, True):
            for module in CONTEXT_ATTENTION_MODULES:
                with self.subTest(module=module.__name__, is_causal=is_causal):
                    actual = module.context_attention(
                        q, k, v, starts, lengths, 1, is_causal
                    )
                    expected = reference(q, k, v, starts, lengths, is_causal)

                    self.assertEqual(
                        (actual.shape, actual.dtype),
                        (q.shape, torch.float32),
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=1e-2, rtol=1e-2
                    )

    def test_head_dim_64_and_128_controls(self):
        for head_dim in (64, 128):
            q, k, v, starts, lengths = make_case(
                torch.float16,
                sequence_lengths=[17, 65],
                num_heads=2,
                head_dim=head_dim,
            )
            for is_causal in (False, True):
                expected = reference(q, k, v, starts, lengths, is_causal)
                for module in CONTEXT_ATTENTION_MODULES:
                    with self.subTest(
                        module=module.__name__,
                        head_dim=head_dim,
                        is_causal=is_causal,
                    ):
                        actual = module.context_attention(
                            q,
                            k,
                            v,
                            starts,
                            lengths,
                            65,
                            is_causal,
                        )
                        self.assertEqual(
                            (actual.shape, actual.dtype),
                            (q.shape, torch.float32),
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=1e-2, rtol=1e-2
                        )

    def test_short_sequences_large_batch_and_bounded_launch(self):
        q, k, v, starts, lengths = make_case(
            torch.bfloat16,
            sequence_lengths=[1] * 257,
            num_heads=2,
            head_dim=16,
        )

        expected = reference(q, k, v, starts, lengths, False)
        for module in CONTEXT_ATTENTION_MODULES:
            actual = module.context_attention(
                q, k, v, starts, lengths, 2048, False
            )
            self.assertEqual(
                (actual.shape, actual.dtype), (q.shape, torch.float32)
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()

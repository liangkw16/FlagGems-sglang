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
    "chunk_state_varlen_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunk_state_varlen.py",
)
ASCEND_MODULE = _load_module(
    "chunk_state_varlen_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "chunk_state_varlen.py",
)
ILUVATAR_MODULE = _load_module(
    "chunk_state_varlen_iluvatar_module",
    BACKEND_ROOT / "_iluvatar" / "ops" / "chunk_state_varlen.py",
)
ENFLAME_MODULE = _load_module(
    "chunk_state_varlen_enflame_module",
    BACKEND_ROOT / "_enflame" / "ops" / "chunk_state_varlen.py",
)
KUNLUN_MODULE = _load_module(
    "chunk_state_varlen_kunlun_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "chunk_state_varlen.py",
)


def reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    _, nheads, headdim = x.shape
    _, _, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups
    states = torch.zeros(
        (batch, nheads, headdim, dstate),
        dtype=chunk_states.dtype,
        device=x.device,
    )
    for batch_index in range(batch):
        start = int(cu_seqlens[batch_index].item())
        end = int(cu_seqlens[batch_index + 1].item())
        chunk = (end - 1) // chunk_size
        chunk_start = chunk * chunk_size
        start_relative = start - chunk_start
        end_relative = end - chunk_start
        for head in range(nheads):
            group = head // ratio
            dA_last = dA_cumsum[head, chunk, end_relative - 1].float()
            x_segment = x[start:end, head].float()
            B_segment = B[start:end, group].float()
            dt_segment = dt[head, chunk, start_relative:end_relative].float()
            dA_segment = dA_cumsum[
                head, chunk, start_relative:end_relative
            ].float()
            scale = torch.exp(dA_last - dA_segment) * dt_segment
            states[batch_index, head] = (
                x_segment.t() @ (B_segment * scale[:, None])
            ).to(chunk_states.dtype)
    return states


def make_case(lengths, dtype, output_dtype, index_dtype=torch.int64):
    chunk_size = 8
    nheads, ngroups = 6, 2
    headdim, dstate = 19, 23
    total_seqlen = sum(lengths)
    nchunks = max((total_seqlen + chunk_size - 1) // chunk_size, 1)
    generator = torch.Generator(device="cuda").manual_seed(
        20260824 + total_seqlen
    )

    x_storage = torch.randn(
        (total_seqlen, nheads, headdim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    B_storage = torch.randn(
        (total_seqlen, ngroups, dstate * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    dt_storage = torch.empty(
        (nheads, nchunks, chunk_size * 2), device="cuda", dtype=dtype
    )
    dA_storage = torch.empty_like(dt_storage)
    x = x_storage[..., ::2]
    B = B_storage[..., 1::2]
    dt = dt_storage[..., ::2]
    dA_cumsum = dA_storage[..., 1::2]
    dt.copy_(torch.rand_like(dt) * 0.1)
    dA_cumsum.copy_((torch.rand_like(dA_cumsum) * 0.1).cumsum(-1))

    cumulative = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=index_dtype,
    )
    cu_storage = torch.empty(
        cumulative.numel() * 2, device="cuda", dtype=index_dtype
    )
    cu_storage[::2] = cumulative
    cu_seqlens = cu_storage[::2]

    chunk_states_storage = torch.randn(
        (nchunks, nheads, headdim, dstate * 2),
        generator=generator,
        device="cuda",
        dtype=output_dtype,
    )
    chunk_states = chunk_states_storage[..., ::2]
    chunk_states.mul_(100.0).add_(7.0)
    return B, x, dt, dA_cumsum, cu_seqlens, chunk_states


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkStateVarlenTest(unittest.TestCase):
    def assert_matches(self, case):
        originals = tuple(tensor.clone() for tensor in case)

        actual = MODULE.chunk_state_varlen(*case)
        expected = reference(*case)

        B, x, _, _, cu_seqlens, chunk_states = case
        self.assertEqual(
            (actual.shape, actual.dtype),
            (
                (
                    cu_seqlens.numel() - 1,
                    x.shape[1],
                    x.shape[2],
                    B.shape[2],
                ),
                chunk_states.dtype,
            ),
        )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        for tensor, original in zip(case, originals):
            torch.testing.assert_close(tensor, original, atol=0.0, rtol=0.0)
        return actual

    def test_gqa_strides_lengths_and_three_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                case = make_case([3, 5, 4], dtype, dtype)

                self.assert_matches(case)

    def test_int32_cu_seqlens_and_chunk_state_values_are_ignored(self):
        case = make_case(
            [1, 7, 2, 6],
            torch.float16,
            torch.float32,
            index_dtype=torch.int32,
        )
        actual = self.assert_matches(case)
        changed_storage = torch.full(
            (7, 4),
            999.0,
            device="cuda",
            dtype=case[-1].dtype,
        )
        changed_chunk_states = changed_storage[:, ::2]

        changed = MODULE.chunk_state_varlen(*case[:-1], changed_chunk_states)

        torch.testing.assert_close(actual, changed, atol=0.0, rtol=0.0)
        for name, module in (
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                vendor_actual = module.chunk_state_varlen(*case)
                vendor_changed = module.chunk_state_varlen(
                    *case[:-1], changed_chunk_states
                )
                self.assertEqual(vendor_actual.dtype, torch.float32)
                torch.testing.assert_close(
                    vendor_actual, vendor_changed, atol=0.0, rtol=0.0
                )

    def test_cross_chunk_single_scale_broadcast(self):
        x = torch.ones((9, 1, 1), device="cuda", dtype=torch.float32)
        B = torch.ones_like(x)
        dt = torch.zeros((1, 2, 8), device="cuda", dtype=torch.float32)
        dt[0, 1, 0] = 1.0
        dA_cumsum = torch.zeros_like(dt)
        cu_seqlens = torch.tensor([0, 9], device="cuda", dtype=torch.int64)
        chunk_states = torch.ones(
            (2, 1, 1, 1), device="cuda", dtype=torch.float32
        )

        expected = reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
        self.assertEqual(expected.item(), 9.0)
        for name, module in (
            ("generic", MODULE),
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(
                    B, x, dt, dA_cumsum, cu_seqlens, chunk_states
                )
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_cross_chunk_broadcast_after_nonzero_start(self):
        x = torch.ones((12, 1, 1), device="cuda", dtype=torch.float32)
        B = torch.ones_like(x)
        dt = torch.zeros((1, 2, 8), device="cuda", dtype=torch.float32)
        dt[0, 0, :3] = 1.0
        dt[0, 1, 3] = 1.0
        dA_cumsum = torch.zeros_like(dt)
        cu_seqlens = torch.tensor([0, 3, 12], device="cuda", dtype=torch.int32)
        chunk_states = torch.empty((1,), device="cuda", dtype=torch.float32)
        case = B, x, dt, dA_cumsum, cu_seqlens, chunk_states
        expected = reference(*case)

        torch.testing.assert_close(
            expected.flatten(), torch.tensor([3.0, 9.0], device="cuda")
        )
        for name, module in (
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(*case)
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_vendor_low_precision_dot_cancellation(self):
        x = torch.tensor(
            [[[64.0]], [[-48.0]]], device="cuda", dtype=torch.float16
        )
        B = torch.ones_like(x)
        dt = torch.tensor(
            [[[2.2734375, 2.86328125]]],
            device="cuda",
            dtype=torch.float16,
        )
        dA_cumsum = torch.tensor(
            [[[0.0, -0.0570068359375]]],
            device="cuda",
            dtype=torch.float16,
        )
        cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
        chunk_states = torch.empty(
            (1, 1, 1, 1), device="cuda", dtype=torch.float32
        )
        case = B, x, dt, dA_cumsum, cu_seqlens, chunk_states
        expected = reference(*case)

        self.assertEqual(expected.item(), 0.0)
        for name, module in (
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(*case)
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_vendor_large_fp32_scale_stays_finite(self):
        x = torch.ones((2, 1, 1), device="cuda", dtype=torch.float32)
        B = torch.ones_like(x)
        dt = torch.ones((1, 1, 2), device="cuda", dtype=torch.float32)
        dA_cumsum = torch.tensor(
            [[[0.0, 12.0]]], device="cuda", dtype=torch.float32
        )
        cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
        chunk_states = torch.empty((1,), device="cuda", dtype=torch.float32)
        case = B, x, dt, dA_cumsum, cu_seqlens, chunk_states
        expected = reference(*case)

        self.assertTrue(torch.isfinite(expected).all())
        for name, module in (
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(*case)
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_leading_empty_sequence(self):
        x = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
        B = torch.ones_like(x)
        dt = torch.ones((1, 1, 8), device="cuda", dtype=torch.float32)
        dA_cumsum = torch.zeros_like(dt)
        cu_seqlens = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int64)
        chunk_states = torch.ones(
            (1, 1, 1, 1), device="cuda", dtype=torch.float32
        )

        expected = reference(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
        torch.testing.assert_close(
            expected.flatten(), torch.tensor([0.0, 1.0], device="cuda")
        )
        for name, module in (
            ("generic", MODULE),
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(
                    B, x, dt, dA_cumsum, cu_seqlens, chunk_states
                )
                torch.testing.assert_close(
                    actual, expected, atol=3e-2, rtol=3e-2
                )

    def test_vendor_all_empty_sequences(self):
        x = torch.empty((0, 1, 1), device="cuda", dtype=torch.float32)
        B = torch.empty_like(x)
        dt = torch.ones((1, 1, 8), device="cuda", dtype=torch.float32)
        dA_cumsum = torch.zeros_like(dt)
        cu_seqlens = torch.tensor([0, 0, 0], device="cuda", dtype=torch.int64)
        chunk_states = torch.empty(
            (1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        )

        for name, module in (
            ("enflame", ENFLAME_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.chunk_state_varlen(
                    B, x, dt, dA_cumsum, cu_seqlens, chunk_states
                )
                self.assertEqual(
                    (actual.shape, actual.dtype),
                    ((2, 1, 1, 1), torch.bfloat16),
                )
                torch.testing.assert_close(
                    actual,
                    torch.zeros_like(actual),
                    atol=0.0,
                    rtol=0.0,
                )

    def test_empty_batch(self):
        B = torch.empty((0, 2, 7), device="cuda", dtype=torch.float16)
        x = torch.empty((0, 4, 5), device="cuda", dtype=torch.float16)
        dt = torch.empty((4, 1, 8), device="cuda", dtype=torch.float16)
        dA_cumsum = torch.empty_like(dt)
        cu_seqlens = torch.tensor([0], device="cuda", dtype=torch.int64)
        chunk_states = torch.ones(
            (1, 4, 5, 7), device="cuda", dtype=torch.bfloat16
        )

        actual = MODULE.chunk_state_varlen(
            B, x, dt, dA_cumsum, cu_seqlens, chunk_states
        )

        self.assertEqual(
            (actual.shape, actual.dtype), ((0, 4, 5, 7), torch.bfloat16)
        )

    def test_vendors_cover_fold_and_fp16_dot(self):
        lengths = [1, 7, 2, 6] * 8

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                case = make_case(lengths, dtype, dtype)

                for name, module in (
                    ("generic", MODULE),
                    ("ascend", ASCEND_MODULE),
                    ("iluvatar", ILUVATAR_MODULE),
                    ("enflame", ENFLAME_MODULE),
                    ("kunlunxin", KUNLUN_MODULE),
                ):
                    with self.subTest(module=name):
                        actual = module.chunk_state_varlen(*case)
                        expected = reference(*case)

                        torch.testing.assert_close(
                            actual, expected, atol=3e-2, rtol=3e-2
                        )


if __name__ == "__main__":
    unittest.main()

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
import math
import unittest
from pathlib import Path

import torch

OPS_PATH = Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops"


def _load_module(name):
    path = OPS_PATH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DECODE_ATTENTION = _load_module("decode_attention")
DECODE_GROUPED_ATTENTION = _load_module("decode_grouped_attention")


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size = kv_indptr.size(0) - 1
    _, query_heads, _ = q.shape
    _, kv_heads, _ = k_buffer.shape
    group_size = query_heads // kv_heads
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    for batch in range(batch_size):
        start = int(kv_indptr[batch].item())
        end = int(kv_indptr[batch + 1].item())
        pages = kv_indices[start:end]
        keys = k_buffer.index_select(0, pages)
        values = v_buffer.index_select(0, pages)
        if kv_heads != query_heads:
            keys = keys.repeat_interleave(group_size, dim=1)
            values = values.repeat_interleave(group_size, dim=1)
        logits = torch.einsum(
            "hd,lhd->hl", q[batch].float(), keys.float()
        ) * float(sm_scale)
        probabilities = torch.softmax(logits - logits.amax(-1, True), dim=-1)
        output[batch] = torch.einsum(
            "hl,lhd->hd", probabilities, values.float()
        )
    return output


def make_case(query_heads, kv_heads, qk_dim, value_dim, lengths, dtype):
    generator = torch.Generator(device="cuda").manual_seed(
        20260824 + query_heads + qk_dim + value_dim
    )
    batch_size = len(lengths)
    total_pages = sum(lengths)
    num_pages = max(total_pages + 5, 1)

    q_base = torch.randn(
        (batch_size, query_heads, qk_dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    k_base = torch.randn(
        (num_pages, kv_heads, qk_dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    v_base = torch.randn(
        (num_pages, kv_heads, value_dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    q = q_base[..., ::2]
    k_buffer = k_base[..., ::2]
    v_buffer = v_base[..., ::2]

    indptr_values = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int64,
    )
    indptr_base = torch.empty(
        indptr_values.numel() * 2, device="cuda", dtype=torch.int64
    )
    indptr_base[::2] = indptr_values
    kv_indptr = indptr_base[::2]

    pages = torch.randperm(num_pages, generator=generator, device="cuda")[
        :total_pages
    ].to(torch.int64)
    indices_base = torch.empty(
        max(total_pages * 2, 1), device="cuda", dtype=torch.int64
    )
    if total_pages:
        indices_base[: total_pages * 2 : 2] = pages
    kv_indices = indices_base[: total_pages * 2 : 2]
    return q, k_buffer, v_buffer, kv_indptr, kv_indices


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class DecodeAttentionCompetitionTest(unittest.TestCase):
    def assert_matches(self, function, case):
        q, k_buffer, v_buffer, kv_indptr, kv_indices = case
        scale = 1.0 / math.sqrt(q.shape[-1])
        snapshots = tuple(
            tensor.clone()
            for tensor in (q, k_buffer, v_buffer, kv_indptr, kv_indices)
        )

        actual = function(q, k_buffer, v_buffer, kv_indptr, kv_indices, scale)
        expected = reference(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, scale
        )

        self.assertEqual(
            (actual.shape, actual.dtype),
            ((q.shape[0], q.shape[1], v_buffer.shape[-1]), torch.float32),
        )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=1e-2)
        for tensor, snapshot in zip(
            (q, k_buffer, v_buffer, kv_indptr, kv_indices), snapshots
        ):
            torch.testing.assert_close(tensor, snapshot, atol=0.0, rtol=0.0)

    def test_mha_strides_variable_lengths_and_value_dim(self):
        case = make_case(4, 4, 33, 17, [1, 35, 65], torch.float16)

        self.assert_matches(DECODE_ATTENTION.decode_attention, case)

    def test_gqa_strides_variable_lengths_and_value_dim(self):
        case = list(make_case(8, 2, 40, 24, [3, 70], torch.bfloat16))
        case[3] = case[3].to(torch.int32)
        case[4] = case[4].to(torch.int32)

        self.assert_matches(
            DECODE_GROUPED_ATTENTION.decode_grouped_attention, tuple(case)
        )

    def test_fp32_mha_and_gqa(self):
        self.assert_matches(
            DECODE_ATTENTION.decode_attention,
            make_case(2, 2, 16, 9, [5], torch.float32),
        )
        self.assert_matches(
            DECODE_GROUPED_ATTENTION.decode_grouped_attention,
            make_case(4, 1, 16, 9, [5], torch.float32),
        )

    def test_grouped_kernel_varied_group_sizes_and_dtypes(self):
        for query_heads, kv_heads, lengths, dtype in (
            (32, 8, [33] * 8, torch.float16),
            (32, 4, [33] * 16, torch.bfloat16),
            (16, 1, [33] * 64, torch.float16),
        ):
            with self.subTest(group_size=query_heads // kv_heads, dtype=dtype):
                case = list(
                    make_case(
                        query_heads,
                        kv_heads,
                        64,
                        64,
                        lengths,
                        dtype,
                    )
                )
                case[:3] = [tensor.contiguous() for tensor in case[:3]]
                self.assert_matches(
                    DECODE_GROUPED_ATTENTION.decode_grouped_attention,
                    tuple(case),
                )

    def test_value_dim_larger_than_qk_dim(self):
        self.assert_matches(
            DECODE_ATTENTION.decode_attention,
            make_case(4, 4, 64, 257, [33, 65], torch.float16),
        )
        self.assert_matches(
            DECODE_GROUPED_ATTENTION.decode_grouped_attention,
            make_case(8, 2, 64, 257, [33, 65], torch.float16),
        )

    def test_empty_batch(self):
        for function, query_heads, kv_heads in (
            (DECODE_ATTENTION.decode_attention, 4, 4),
            (
                DECODE_GROUPED_ATTENTION.decode_grouped_attention,
                8,
                2,
            ),
        ):
            with self.subTest(function=function.__name__):
                q = torch.empty(
                    (0, query_heads, 16), device="cuda", dtype=torch.float16
                )
                k_buffer = torch.empty(
                    (1, kv_heads, 16), device="cuda", dtype=torch.float16
                )
                v_buffer = torch.empty(
                    (1, kv_heads, 9), device="cuda", dtype=torch.float16
                )
                kv_indptr = torch.tensor([0], device="cuda", dtype=torch.int64)
                kv_indices = torch.empty(0, device="cuda", dtype=torch.int64)

                actual = function(
                    q,
                    k_buffer,
                    v_buffer,
                    kv_indptr,
                    kv_indices,
                    0.25,
                )

                self.assertEqual(
                    (actual.shape, actual.dtype),
                    ((0, query_heads, 9), torch.float32),
                )


if __name__ == "__main__":
    unittest.main()

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
    / "embedding_lora_a.py"
)
SPEC = importlib.util.spec_from_file_location(
    "embedding_lora_a_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(
    input_ids, weights, batch_info, vocab_size, extra_embeddings=None
):
    output = torch.zeros(
        input_ids.shape[0],
        weights.shape[1],
        dtype=weights.dtype,
        device=weights.device,
    )
    for batch_id in range(batch_info.bs):
        start = int(batch_info.seg_indptr[batch_id].item())
        end = int(batch_info.seg_indptr[batch_id + 1].item())
        if start == end:
            continue
        weight_index = int(batch_info.weight_indices[batch_id].item())
        rank = int(batch_info.lora_ranks[weight_index].item())
        if rank == 0:
            continue

        tokens = input_ids[start:end].long()
        is_extra = tokens >= vocab_size
        clamped = tokens.clamp(max=vocab_size - 1)
        output[start:end, :rank] = weights[weight_index, :rank, clamped].t()
        if extra_embeddings is not None and bool(is_extra.any()):
            extra_indices = (tokens - vocab_size).clamp(min=0)
            extra_values = extra_embeddings[weight_index, extra_indices, :rank]
            output[start:end, :rank] = torch.where(
                is_extra.unsqueeze(-1),
                extra_values,
                output[start:end, :rank],
            )
    return output


def batch_info(seg_indptr, weight_indices, lora_ranks):
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
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class EmbeddingLoraATest(unittest.TestCase):
    def test_segments_rank_zero_and_dtypes_match_reference(self):
        tolerances = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }
        info = batch_info([0, 2, 2, 5, 7], [0, 1, 1, 2], [3, 0, 5])
        input_ids = torch.tensor(
            [1, 7, 2, 6, 7, 0, 5], device="cuda", dtype=torch.int64
        )

        for dtype, tolerance in tolerances.items():
            with self.subTest(dtype=dtype):
                weights = (
                    torch.arange(3 * 5 * 6, device="cuda")
                    .reshape(3, 5, 6)
                    .to(dtype)
                    / 100
                )
                input_before = input_ids.clone()
                weights_before = weights.clone()
                info_before = tuple(
                    value.clone()
                    for value in (
                        info.seg_lens,
                        info.seg_indptr,
                        info.weight_indices,
                        info.lora_ranks,
                    )
                )

                actual = MODULE.embedding_lora_a(input_ids, weights, info, 6)
                expected = reference(input_ids, weights, info, 6)

                self.assertEqual(actual.shape, (7, 5))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    input_ids, input_before, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    weights, weights_before, atol=0, rtol=0
                )
                for value, before in zip(
                    (
                        info.seg_lens,
                        info.seg_indptr,
                        info.weight_indices,
                        info.lora_ranks,
                    ),
                    info_before,
                ):
                    torch.testing.assert_close(value, before, atol=0, rtol=0)

    def test_extra_embeddings_replace_extra_tokens(self):
        info = batch_info([0, 3, 4], [1, 0], [2, 4])
        input_ids = torch.tensor([0, 6, 7, 5], device="cuda")
        weights = (
            torch.arange(2 * 4 * 6, device="cuda", dtype=torch.float32)
            .reshape(2, 4, 6)
            .div(10)
        )
        extra_embeddings = (
            torch.arange(2 * 2 * 4, device="cuda", dtype=torch.float32)
            .reshape(2, 2, 4)
            .add(100)
        )
        inputs_before = (
            input_ids.clone(),
            weights.clone(),
            extra_embeddings.clone(),
        )

        actual = MODULE.embedding_lora_a(
            input_ids, weights, info, 6, extra_embeddings
        )
        expected = reference(input_ids, weights, info, 6, extra_embeddings)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        for value, before in zip(
            (input_ids, weights, extra_embeddings), inputs_before
        ):
            torch.testing.assert_close(value, before, atol=0, rtol=0)

    def test_rank_block_tail_and_noncontiguous_strides(self):
        info = batch_info([0, 3], [0], [130])
        input_ids = torch.tensor(
            [1, 99, 4, 99, 7], device="cuda", dtype=torch.int64
        )[::2]
        weights = torch.arange(
            130 * 16, device="cuda", dtype=torch.float32
        ).reshape(1, 130, 16)[:, :, ::2]
        self.assertFalse(input_ids.is_contiguous())
        self.assertFalse(weights.is_contiguous())

        actual = MODULE.embedding_lora_a(input_ids, weights, info, 8)
        expected = reference(input_ids, weights, info, 8)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

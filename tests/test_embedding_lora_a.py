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
    "embedding_lora_a_module",
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "embedding_lora_a.py",
)
ASCEND_MODULE = _load_module(
    "embedding_lora_a_ascend_module",
    BACKEND_ROOT / "_ascend" / "ops" / "embedding_lora_a.py",
)
KUNLUN_MODULE = _load_module(
    "embedding_lora_a_kunlunxin_module",
    BACKEND_ROOT / "_kunlunxin" / "ops" / "embedding_lora_a.py",
)
ENFLAME_MODULE = _load_module(
    "embedding_lora_a_enflame_module",
    BACKEND_ROOT / "_enflame" / "ops" / "embedding_lora_a.py",
)


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

    def test_rank_block_boundaries_and_metadata_strides(self):
        def strided(values):
            storage = torch.empty(
                len(values) * 2, device="cuda", dtype=torch.int32
            )
            storage[::2] = torch.tensor(
                values, device="cuda", dtype=torch.int32
            )
            return storage[::2]

        input_ids = torch.tensor(
            [1, 99, 4, 99], device="cuda", dtype=torch.int32
        )[::2]
        for rank in (127, 128, 129):
            with self.subTest(rank=rank):
                info = SimpleNamespace(
                    bs=1,
                    max_len=2,
                    seg_lens=strided([2]),
                    seg_indptr=strided([0, 2]),
                    weight_indices=strided([0]),
                    lora_ranks=strided([rank]),
                )
                weights = torch.arange(
                    rank * 16, device="cuda", dtype=torch.float32
                ).reshape(1, rank, 16)[:, :, ::2]
                originals = (
                    input_ids.clone(),
                    weights.clone(),
                    info.seg_lens.clone(),
                    info.seg_indptr.clone(),
                    info.weight_indices.clone(),
                    info.lora_ranks.clone(),
                )

                actual = MODULE.embedding_lora_a(
                    input_ids, weights, info, vocab_size=8
                )
                expected = reference(input_ids, weights, info, vocab_size=8)

                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )
                for value, before in zip(
                    (
                        input_ids,
                        weights,
                        info.seg_lens,
                        info.seg_indptr,
                        info.weight_indices,
                        info.lora_ranks,
                    ),
                    originals,
                ):
                    torch.testing.assert_close(value, before, atol=0, rtol=0)
                self.assertEqual(info.seg_indptr.stride(), (2,))

    def test_seg_indptr_is_authoritative_and_empty_metadata_is_ignored(self):
        info = SimpleNamespace(
            bs=3,
            max_len=2,
            seg_lens=torch.tensor([1, 0, 1], device="cuda"),
            seg_indptr=torch.tensor([0, 2, 2, 3], device="cuda"),
            weight_indices=torch.tensor(
                [0, 1 << 28, 0], device="cuda", dtype=torch.int64
            ),
            lora_ranks=torch.tensor([2], device="cuda", dtype=torch.int64),
        )
        input_ids = torch.tensor([1, 2, 3], device="cuda")
        weights = torch.arange(
            2 * 8, device="cuda", dtype=torch.float32
        ).reshape(1, 2, 8)

        actual = MODULE.embedding_lora_a(input_ids, weights, info, 8)
        expected = reference(input_ids, weights, info, 8)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_vendors_cover_token_fold_and_configs(self):
        torch.manual_seed(20260824)
        vocab_size = 1024
        seg_indptr = [0]
        import random

        random.seed(20260824)
        for _ in range(8):
            seg_indptr.append(seg_indptr[-1] + random.randint(9000, 16384))
        info = batch_info(seg_indptr, list(range(8)), [64] * 8)
        self.assertGreater(info.max_len, 65535 // 8)
        input_ids = torch.randint(
            0, vocab_size, (seg_indptr[-1],), device="cuda"
        ).to(torch.int32)
        weights = torch.randn(
            (8, 64, vocab_size), device="cuda", dtype=torch.float16
        )
        expected = reference(input_ids, weights, info, vocab_size)

        for name, module in (
            ("generic", MODULE),
            ("ascend", ASCEND_MODULE),
            ("kunlunxin", KUNLUN_MODULE),
            ("enflame", ENFLAME_MODULE),
        ):
            with self.subTest(module=name):
                actual = module.embedding_lora_a(
                    input_ids, weights, info, vocab_size
                )
                torch.testing.assert_close(
                    actual, expected, atol=0.0, rtol=0.0
                )


if __name__ == "__main__":
    unittest.main()

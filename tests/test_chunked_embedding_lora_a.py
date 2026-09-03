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

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunked_embedding_lora_a.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunked_embedding_lora_a_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


class BatchInfo:
    def __init__(self, seg_indptr, weight_indices, lora_ranks, permutation, bs):
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices
        self.lora_ranks = lora_ranks
        self.permutation = permutation
        self.bs = bs


def reference(input_ids, weights, batch_info, vocab_size):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = weights.new_zeros(S, rank)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        rows = permutation[start:end].long()
        tokens = input_ids[rows].long()
        out[rows, :r] = weights[w_idx, :r, tokens].t()

    return out


def make_case(
    seg_lens,
    ranks,
    max_rank=256,
    vocab_size=1024,
    num_lora=None,
    dtype=torch.float32,
    id_dtype=torch.int64,
    seed=0,
    sentinel_empty_widx=False,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_lora = num_lora or max(len(ranks), 1)
    S = sum(seg_lens)
    weights = torch.randn(
        num_lora, max_rank, vocab_size, dtype=dtype, generator=g
    ).cuda()
    input_ids = torch.randint(0, vocab_size, (S,), dtype=id_dtype, generator=g).cuda()
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()),
        dtype=id_dtype,
    ).cuda()
    weight_indices = torch.randint(
        0, num_lora, (len(seg_lens),), dtype=id_dtype, generator=g
    ).cuda()
    if sentinel_empty_widx:
        for b, length in enumerate(seg_lens):
            if length == 0:
                weight_indices[b] = 10**6  # out-of-range sentinel
    lora_ranks = torch.tensor(ranks, dtype=id_dtype).cuda()
    permutation = torch.randperm(S, generator=g).to(id_dtype).cuda()
    batch_info = BatchInfo(
        seg_indptr, weight_indices, lora_ranks, permutation, len(seg_lens)
    )
    return input_ids, weights, batch_info, None


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedEmbeddingLoraATest(unittest.TestCase):
    def _check(self, input_ids, weights, batch_info):
        ids_snapshot = input_ids.clone()
        weights_snapshot = weights.clone()
        actual = MODULE.chunked_embedding_lora_a(
            input_ids, weights, batch_info, weights.shape[2]
        )
        expected = reference(input_ids, weights, batch_info, weights.shape[2])
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[weights.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(input_ids, ids_snapshot)
        torch.testing.assert_close(weights, weights_snapshot)
        return actual

    def test_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = make_case([12, 7, 20, 3], [32, 0, 128, 16], dtype=dtype)
                self._check(*args[:3])

    def test_id_dtypes(self):
        for id_dtype in (torch.int32, torch.int64):
            with self.subTest(id_dtype=id_dtype):
                args = make_case([5, 9, 2], [64, 8, 200], id_dtype=id_dtype)
                self._check(*args[:3])

    def test_rank_boundaries(self):
        for ranks in ([0], [1], [127], [128], [129], [200], [256]):
            with self.subTest(ranks=ranks):
                args = make_case([4, 11], ranks, max_rank=256, seed=1)
                self._check(*args[:3])

    def test_empty_segments_with_sentinel_widx(self):
        # Empty segments carry out-of-range adapter sentinels; a kernel
        # that reads their metadata before checking the span crashes or
        # corrupts (the T17 S0/S1 lesson).
        args = make_case(
            [0, 8, 0, 0, 5, 0, 3],
            [16, 32, 64, 8, 0, 4, 128],
            sentinel_empty_widx=True,
            seed=2,
        )
        self._check(*args[:3])

    def test_all_ranks_zero(self):
        args = make_case([6, 6], [0, 0], seed=3)
        out = self._check(*args[:3])
        self.assertTrue(torch.all(out == 0))

    def test_single_segment_and_single_token(self):
        args = make_case([1], [64], seed=4)
        self._check(*args[:3])
        args = make_case([17], [128], seed=5)
        self._check(*args[:3])

    def test_many_segments(self):
        seg_lens = [1 + (i * 7) % 13 for i in range(40)]
        ranks = [(i * 37) % 200 for i in range(40)]
        args = make_case(seg_lens, ranks, seed=6)
        self._check(*args[:3])

    def test_columns_beyond_rank_stay_zero(self):
        # Rows written with an effective rank r < max_rank must keep
        # every column >= r at exactly zero (the T17 trap).
        input_ids, weights, batch_info, _ = make_case(
            [9, 9, 9], [64, 0, 200], max_rank=256, seed=7
        )
        # Pin adapters deterministically: segment b uses adapter b.
        batch_info.weight_indices = torch.arange(
            3, dtype=batch_info.weight_indices.dtype, device="cuda"
        )
        out = self._check(input_ids, weights, batch_info)
        seg = batch_info.seg_indptr.cpu()
        for b, r in enumerate((64, 0, 200)):
            rows = batch_info.permutation[seg[b] : seg[b + 1]]
            if r < 256:
                self.assertTrue(torch.all(out[rows, r:] == 0))

    def test_non_contiguous_weights(self):
        args = make_case([8, 8], [64, 32], max_rank=256, vocab_size=512)
        input_ids, weights, batch_info, _ = args
        big = torch.randn(
            weights.shape[0],
            weights.shape[1],
            weights.shape[2] * 2,
            device="cuda",
        )
        weights_nc = big[:, :, ::2]
        self.assertFalse(weights_nc.is_contiguous())
        self._check(input_ids, weights_nc, batch_info)

    def test_empty_batch(self):
        args = make_case([], [], seed=8)
        input_ids, weights, batch_info, _ = args
        out = MODULE.chunked_embedding_lora_a(
            input_ids, weights, batch_info, weights.shape[2]
        )
        self.assertEqual(out.shape, (0, weights.shape[1]))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedEmbeddingLoraAVariantsTest(unittest.TestCase):
    """Run the core matrix against every backend variant (generic +
    enflame i32 route), so vendor files cannot ship numerically broken."""

    MODULES = load_operator_modules("chunked_embedding_lora_a")

    def test_variants_match_reference(self):
        cases = [
            ([12, 7, 20, 3], [32, 0, 128, 16], 256, 1024),
            ([0, 8, 0, 0, 5, 0, 3], [16, 32, 64, 8, 0, 4, 128], 256, 512),
            ([1], [64], 128, 256),
        ]
        for seg_lens, ranks, max_rank, vocab in cases:
            input_ids, weights, batch_info, _ = make_case(
                seg_lens, ranks, max_rank=max_rank, vocab_size=vocab,
                sentinel_empty_widx=True, seed=11,
            )
            ref = reference(input_ids, weights, batch_info, vocab)
            for name, module in self.MODULES:
                with self.subTest(module=name, seg_lens=seg_lens):
                    out = module.chunked_embedding_lora_a(
                        input_ids, weights, batch_info, vocab
                    )
                    torch.testing.assert_close(
                        out, ref, atol=1e-5, rtol=1e-5
                    )

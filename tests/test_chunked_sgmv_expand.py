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
    / "chunked_sgmv_expand.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunked_sgmv_expand_module", MODULE_PATH
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
    def __init__(
        self,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        permutation,
        bs,
    ):
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices
        self.lora_ranks = lora_ranks
        self.scalings = scalings
        self.permutation = permutation
        self.bs = bs


def reference(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    out = base_output.clone().float()
    n_slices = slice_offsets.numel() - 1
    r = weights.shape[-1]

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    scalings = batch_info.scalings
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        scaling = float(scalings[w_idx].item())
        rows = permutation[start:end].long()

        x_seg = x[rows].float()
        for i in range(n_slices):
            o_start = int(slice_offsets[i].item())
            o_end = int(slice_offsets[i + 1].item())
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = weights[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)


def make_case(
    seg_lens,
    num_lora,
    slice_widths,
    rank,
    dtype=torch.float32,
    seed=0,
    base_fill=1.0,
):
    g = torch.Generator().manual_seed(seed)
    S = sum(seg_lens)
    n_slices = len(slice_widths)
    total_out = sum(slice_widths)
    x = torch.randn(S, n_slices * rank, dtype=dtype, generator=g).cuda()
    weights = (
        torch.randn(num_lora, total_out, rank, dtype=dtype, generator=g)
        .cuda()
        .to(dtype)
    )
    base_output = (
        torch.full((S, total_out), base_fill, dtype=dtype).cuda().to(dtype)
    )
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()),
        dtype=torch.int64,
    ).cuda()
    weight_indices = torch.randint(
        0, num_lora, (len(seg_lens),), dtype=torch.int64, generator=g
    ).cuda()
    lora_ranks = (
        torch.randint(0, 2, (num_lora,), dtype=torch.int64, generator=g).cuda()
        * rank
    )  # 0 or full rank
    scalings = torch.randn(num_lora, dtype=dtype, generator=g).cuda()
    permutation = torch.randperm(S, generator=g).cuda()
    slice_offsets = torch.tensor(
        [0] + list(torch.tensor(slice_widths).cumsum(0).tolist()),
        dtype=torch.int64,
    ).cuda()
    batch_info = BatchInfo(
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        permutation,
        len(seg_lens),
    )
    return x, weights, batch_info, slice_offsets, base_output


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedSgmvExpandTest(unittest.TestCase):
    def _check(self, x, weights, batch_info, slice_offsets, base_output):
        snapshots = [t.clone() for t in (x, weights, base_output)]
        max_slice_size = int(
            (slice_offsets[1:] - slice_offsets[:-1]).max().item()
        )
        actual = MODULE.chunked_sgmv_expand(
            x, weights, batch_info, slice_offsets, max_slice_size, base_output
        )
        expected = reference(
            x, weights, batch_info, slice_offsets, max_slice_size, base_output
        )
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[base_output.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for tensor, snapshot in zip((x, weights, base_output), snapshots):
            torch.testing.assert_close(tensor, snapshot)
        return actual

    def test_dtypes_equal_slice(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = make_case([16, 32, 8], 4, [128, 128], 32, dtype=dtype)
                self._check(*args)

    def test_unequal_slice_widths(self):
        args = make_case([32, 16], 3, [65, 80, 129], 16, seed=1)
        self._check(*args)

    def test_rank_sizes(self):
        for rank in (16, 32, 64, 8):
            with self.subTest(rank=rank):
                args = make_case([20, 20], 3, [128, 64], rank, seed=2)
                self._check(*args)

    def test_empty_segments_and_zero_ranks(self):
        # Empty segments carry sentinel adapter indices; rank-0 adapters
        # contribute nothing (rows keep base values).
        args = make_case([0, 12, 0, 12, 0], 3, [64, 64], 32, seed=3)
        x, weights, batch_info, slice_offsets, base_output = args
        batch_info.weight_indices[0] = 10**6
        batch_info.weight_indices[2] = 10**6
        batch_info.weight_indices[4] = 10**6
        self._check(x, weights, batch_info, slice_offsets, base_output)

    def test_all_rank_zero(self):
        args = make_case([10, 10], 2, [64, 64], 32, seed=4)
        args[2].lora_ranks[:] = 0
        self._check(*args)
        # Output equals base_output untouched.

    def test_single_segment_single_token(self):
        args = make_case([1], 1, [32], 16, seed=5)
        self._check(*args)

    def test_identity_permutation(self):
        args = make_case([24, 24], 2, [64, 32], 16, seed=6)
        x, weights, batch_info, slice_offsets, base_output = args
        batch_info.permutation = torch.arange(
            x.shape[0], dtype=torch.int64, device="cuda"
        )
        self._check(x, weights, batch_info, slice_offsets, base_output)

    def test_base_output_untouched_for_rank_zero(self):
        args = make_case([12, 12], 2, [64, 64], 32, seed=7)
        x, weights, batch_info, slice_offsets, base_output = args
        batch_info.lora_ranks[:] = 0
        max_slice = 64
        out = MODULE.chunked_sgmv_expand(
            x, weights, batch_info, slice_offsets, max_slice, base_output
        )
        torch.testing.assert_close(out, base_output)
        self.assertFalse(out is base_output)

    def test_empty_batch(self):
        args = make_case([], 1, [64], 32, seed=8)
        x, weights, batch_info, slice_offsets, base_output = args
        out = MODULE.chunked_sgmv_expand(
            x, weights, batch_info, slice_offsets, 64, base_output
        )
        self.assertEqual(out.shape, base_output.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedSgmvExpandVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + enflame)."""

    MODULES = load_operator_modules("chunked_sgmv_expand")

    def test_vendor_multi_k_pointer_stride(self):
        # Public-entry regression replaces the retired private GEMM entry.
        for rank in (65, 193):
            x, weights, info, offsets, base = make_case(
                [3, 0, 2], 2, [17, 33], rank, seed=92
            )
            info.lora_ranks.fill_(rank)
            expected = reference(x, weights, info, offsets, 33, base)
            for name, module in self.MODULES:
                with self.subTest(module=name, rank=rank):
                    actual = module.chunked_sgmv_expand(
                        x, weights, info, offsets, 33, base
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=1e-4, rtol=1e-4
                    )

    def test_route_mutation_partial_and_grid_tail(self):
        for lengths, widths in (
            ([0, 3, 0, 2, 0], [15, 16, 17]),
            ([129], [8193]),
        ):
            x, weights, info, offsets, base = make_case(
                lengths, 2, widths, 8, seed=94
            )
            info.lora_ranks.fill_(8)
            for _ in range(2):
                info.weight_indices.copy_(1 - info.weight_indices)
                expected = reference(
                    x, weights, info, offsets, max(widths), base
                )
                for name, module in self.MODULES:
                    with self.subTest(module=name, lengths=lengths):
                        actual = module.chunked_sgmv_expand(
                            x, weights, info, offsets, max(widths), base
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=1e-4, rtol=1e-4
                        )
            if len(lengths) > 1:
                info.seg_indptr[-2:] = 3
                expected = reference(
                    x, weights, info, offsets, max(widths), base
                )
                for name, module in self.MODULES:
                    with self.subTest(module=name, partial=True):
                        actual = module.chunked_sgmv_expand(
                            x, weights, info, offsets, max(widths), base
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=1e-4, rtol=1e-4
                        )

    def test_skewed_segments_and_empty_prefix(self):
        for lengths in (
            [1024] + [1] * 31,
            [0] * 31 + [65],
            [0, 1, 0, 64, 0, 65],
            [0] * 64,
            [0] * 4096 + [1],
        ):
            x, w, info, offsets, base = make_case(
                lengths, 3, [17, 33], 16, seed=98
            )
            info.lora_ranks.fill_(16)
            expected = reference(x, w, info, offsets, 33, base)
            for hint in (False, True):
                if hint:
                    info.max_len = max(lengths)
                actual = MODULE.chunked_sgmv_expand(
                    x, w, info, offsets, 33, base
                )
                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )

    def test_int32_metadata_and_strides(self):
        x, weights, info, offsets, base = make_case(
            [0, 7, 2, 0], 3, [17, 33], 33, seed=95
        )
        info.lora_ranks.fill_(33)

        def strided(t):
            storage = torch.empty(
                (*t.shape[:-1], t.shape[-1] * 2),
                dtype=t.dtype,
                device=t.device,
            )
            storage[..., ::2] = t
            return storage[..., ::2]

        x, weights, base = (strided(t) for t in (x, weights, base))
        for name in (
            "seg_indptr",
            "weight_indices",
            "lora_ranks",
            "permutation",
        ):
            setattr(info, name, strided(getattr(info, name).to(torch.int32)))
        info.scalings = strided(info.scalings)
        offsets = strided(offsets.to(torch.int32))
        expected = reference(x, weights, info, offsets, 33, base)
        snapshots = [t.clone() for t in (x, weights, base)]
        for name, module in self.MODULES:
            with self.subTest(module=name):
                actual = module.chunked_sgmv_expand(
                    x, weights, info, offsets, 33, base
                )
                torch.testing.assert_close(
                    actual, expected, atol=1e-4, rtol=1e-4
                )
                for tensor, snapshot in zip((x, weights, base), snapshots):
                    torch.testing.assert_close(
                        tensor, snapshot, atol=0, rtol=0
                    )

    def test_rank_beyond_single_k_tile(self):
        # Large ranks must use several bounded K tiles. The next rank must
        # advance B along its rank stride, not its output-column stride.
        for rank in (127, 128, 129, 511, 512, 513):
            args = make_case([33, 0, 65], 2, [17, 33], rank, seed=47)
            x, weights, info, offsets, base = args
            info.lora_ranks[:] = 1  # nonzero means use the stored rank
            expected = reference(x, weights, info, offsets, 33, base)
            for name, module in self.MODULES:
                with self.subTest(module=name, rank=rank):
                    actual = module.chunked_sgmv_expand(
                        x, weights, info, offsets, 33, base
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=1e-4, rtol=1e-4
                    )

    def test_variants_match_reference(self):
        cases = [
            ([16, 32, 8], 4, [128, 128], 32, torch.float32),
            ([0, 12, 0, 12, 0], 3, [65, 80, 129], 16, torch.float32),
            ([20, 20], 3, [128, 64], 16, torch.bfloat16),
            # Keep historical rank regressions; the small direct-kernel
            # test separately exercises the corrected rank-stride advance.
            ([24, 12], 3, [128, 64], 64, torch.float32),
            ([24, 12], 3, [128, 64], 96, torch.float32),
            ([24, 12], 3, [128, 64], 128, torch.float32),
        ]
        for seg_lens, num_lora, widths, rank, dtype in cases:
            x, weights, batch_info, slice_offsets, base_output = make_case(
                seg_lens, num_lora, widths, rank, dtype=dtype, seed=21
            )
            max_slice = max(widths)
            ref = reference(
                x, weights, batch_info, slice_offsets, max_slice, base_output
            )
            for name, module in self.MODULES:
                with self.subTest(module=name, seg_lens=seg_lens):
                    out = module.chunked_sgmv_expand(
                        x,
                        weights,
                        batch_info,
                        slice_offsets,
                        max_slice,
                        base_output,
                    )
                    atol, rtol = TOLERANCES[base_output.dtype]
                    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


RELEASE_REQUIRED_TESTS = [
    "ChunkedSgmvExpandTest.test_dtypes_equal_slice",
    "ChunkedSgmvExpandTest.test_unequal_slice_widths",
    "ChunkedSgmvExpandTest.test_rank_sizes",
    "ChunkedSgmvExpandTest.test_empty_segments_and_zero_ranks",
    "ChunkedSgmvExpandTest.test_all_rank_zero",
    "ChunkedSgmvExpandTest.test_single_segment_single_token",
    "ChunkedSgmvExpandTest.test_identity_permutation",
    "ChunkedSgmvExpandTest.test_base_output_untouched_for_rank_zero",
    "ChunkedSgmvExpandTest.test_empty_batch",
    "ChunkedSgmvExpandVariantsTest.test_variants_match_reference",
    "ChunkedSgmvExpandVariantsTest.test_rank_beyond_single_k_tile",
    "ChunkedSgmvExpandVariantsTest.test_vendor_multi_k_pointer_stride",
    "ChunkedSgmvExpandVariantsTest.test_route_mutation_partial_and_grid_tail",
    "ChunkedSgmvExpandVariantsTest.test_int32_metadata_and_strides",
    "ChunkedSgmvExpandVariantsTest.test_skewed_segments_and_empty_prefix",
]


if __name__ == "__main__":
    unittest.main()

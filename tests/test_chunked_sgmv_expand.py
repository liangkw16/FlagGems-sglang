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
SPEC = importlib.util.spec_from_file_location("chunked_sgmv_expand_module", MODULE_PATH)
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


def reference(x, weights, batch_info, slice_offsets, max_slice_size, base_output):
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
    base_output = torch.full((S, total_out), base_fill, dtype=dtype).cuda().to(dtype)
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()),
        dtype=torch.int64,
    ).cuda()
    weight_indices = torch.randint(
        0, num_lora, (len(seg_lens),), dtype=torch.int64, generator=g
    ).cuda()
    lora_ranks = (
        torch.randint(0, 2, (num_lora,), dtype=torch.int64, generator=g).cuda() * rank
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
        max_slice_size = int((slice_offsets[1:] - slice_offsets[:-1]).max().item())
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


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedSgmvExpandVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + enflame)."""

    MODULES = load_operator_modules("chunked_sgmv_expand")

    def test_variants_match_reference(self):
        cases = [
            ([16, 32, 8], 4, [128, 128], 32, torch.float32),
            ([0, 12, 0, 12, 0], 3, [65, 80, 129], 16, torch.float32),
            ([20, 20], 3, [128, 64], 16, torch.bfloat16),
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

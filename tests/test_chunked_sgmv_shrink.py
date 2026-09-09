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

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "chunked_sgmv_shrink.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunked_sgmv_shrink_module", MODULE_PATH
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
    def __init__(self, seg_indptr, weight_indices, permutation, bs):
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices
        self.permutation = permutation
        self.bs = bs


def reference(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    out = x.new_zeros(S, N)
    for b in range(batch_info.bs):
        start = int(batch_info.seg_indptr[b].item())
        end = int(batch_info.seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(batch_info.weight_indices[b].item())
        rows = batch_info.permutation[start:end].long()
        x_seg = x[rows].float()
        w = weights[w_idx].float()
        out[rows] = (x_seg @ w.t()).to(x.dtype)
    return out


def reference_f64(x, weights, batch_info, num_slices=1):
    """float64 reference, for separating fp32 accumulation order from bugs."""
    S, K = x.shape
    N = weights.shape[1]
    out = x.new_zeros(S, N, dtype=torch.float64)
    for b in range(batch_info.bs):
        start = int(batch_info.seg_indptr[b].item())
        end = int(batch_info.seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(batch_info.weight_indices[b].item())
        rows = batch_info.permutation[start:end].long()
        out[rows] = x[rows].double() @ weights[w_idx].double().t()
    return out


def make_case(seg_lens, num_lora, K, N, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = sum(seg_lens)
    x = torch.randn(S, K, dtype=dtype, generator=g).cuda().to(dtype)
    weights = (
        torch.randn(num_lora, N, K, dtype=dtype, generator=g).cuda().to(dtype)
    )
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0).tolist()),
        dtype=torch.int64,
    ).cuda()
    weight_indices = torch.randint(
        0, num_lora, (len(seg_lens),), dtype=torch.int64, generator=g
    ).cuda()
    permutation = torch.randperm(S, generator=g).cuda()
    batch_info = BatchInfo(
        seg_indptr, weight_indices, permutation, len(seg_lens)
    )
    return x, weights, batch_info


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedSgmvShrinkTest(unittest.TestCase):
    def _check(self, x, weights, batch_info, num_slices=1):
        x_snap = x.clone()
        w_snap = weights.clone()
        actual = MODULE.chunked_sgmv_shrink(x, weights, batch_info, num_slices)
        expected = reference(x, weights, batch_info, num_slices)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[x.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(x, x_snap)
        torch.testing.assert_close(weights, w_snap)
        return actual

    def test_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = make_case([16, 32, 8], 4, 512, 128, dtype=dtype)
                self._check(*args)

    def test_shapes(self):
        for seg_lens, K, N in (
            ([32], 1024, 64),
            ([8, 8, 8, 8], 512, 256),
            ([64, 32, 16], 2048, 32),
            ([1], 128, 16),
            ([0, 12, 0, 12, 0], 512, 128),
            ([24, 12], 65, 80),  # non-pow2 K and N
            ([63, 64, 65, 256], 128, 32),  # segment spans multiple M tiles
        ):
            with self.subTest(seg_lens=seg_lens, K=K, N=N):
                args = make_case(seg_lens, 3, K, N)
                self._check(*args)

    def test_identity_permutation(self):
        x, weights, bi = make_case([24, 24], 2, 512, 128)
        bi.permutation = torch.arange(
            x.shape[0], dtype=torch.int64, device="cuda"
        )
        self._check(x, weights, bi)

    def test_tile_geometry_axes(self):
        # e5 derives BLOCK_N from N (the output/rank axis) and walks the
        # reduction axis K with BLOCK_K, inverting e4's mapping. Pin both
        # axes around every tile boundary the new selection can pick:
        # BLOCK_N floors at 16, doubles up to 128, then must tile; BLOCK_K
        # tops out at 128, so K needs values just under, equal to and just
        # over each, plus non-pow2 tails.
        #
        # This test targets tile masking and coverage, which fail loudly
        # (whole rows/columns wrong, O(1) relative error). It deliberately
        # includes K=4096, where summing 4096 fp32 products in a different
        # order than torch.matmul costs slightly more than the task's flat
        # atol=1e-4 on a handful of elements -- the committed e4 kernel
        # shows byte-identical error there, so a flat bound would flag
        # reference accumulation order rather than a kernel defect. Compare
        # against a float64 reference and scale the bound with sqrt(K), the
        # expected growth of fp32 accumulation error.
        for K in (1, 127, 128, 129, 255, 256, 257, 384, 4096):
            for N in (1, 15, 16, 17, 31, 32, 33, 128, 129, 200):
                with self.subTest(K=K, N=N):
                    x, weights, bi = make_case(
                        [17, 48], 2, K, N, seed=K * 1000 + N
                    )
                    x_snap = x.clone()
                    w_snap = weights.clone()
                    actual = MODULE.chunked_sgmv_shrink(x, weights, bi)
                    expected = reference_f64(x, weights, bi)
                    self.assertEqual(actual.shape, expected.shape)
                    bound = 1e-4 * max(1.0, math.sqrt(K) / 16.0)
                    torch.testing.assert_close(
                        actual.double(),
                        expected,
                        atol=bound,
                        rtol=bound,
                    )
                    # A masking or coverage bug cannot hide inside the
                    # numeric bound: no element may be grossly wrong.
                    scale = expected.abs().amax().clamp(min=1e-6)
                    self.assertLess(
                        (actual.double() - expected).abs().amax().item(),
                        0.01 * scale.item(),
                    )
                    torch.testing.assert_close(x, x_snap)
                    torch.testing.assert_close(weights, w_snap)

    def test_empty_batch(self):
        x, weights, bi = make_case([], 1, 512, 128)
        out = MODULE.chunked_sgmv_shrink(x, weights, bi)
        self.assertEqual(out.shape, (0, 128))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkedSgmvShrinkVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("chunked_sgmv_shrink")

    def test_pipeline_long_segment_dtypes(self):
        for dtype in TOLERANCES:
            args = make_case(
                [63, 64, 65, 256, 257], 3, 129, 33, dtype=dtype, seed=48
            )
            expected = reference(*args)
            for name, module in self.MODULES:
                with self.subTest(module=name, dtype=dtype):
                    actual = module.chunked_sgmv_shrink(*args)
                    atol, rtol = TOLERANCES[dtype]
                    torch.testing.assert_close(
                        actual, expected, atol=atol, rtol=rtol
                    )

    def test_variants_match_reference(self):
        for seg_lens, K, N in (
            ([16, 32, 8], 512, 128),
            ([0, 12, 0, 12, 0], 512, 128),
            ([24, 12], 65, 80),
            ([63, 64, 65, 256], 128, 32),
        ):
            x, weights, bi = make_case(seg_lens, 3, K, N)
            ref = reference(x, weights, bi)
            for name, module in self.MODULES:
                with self.subTest(module=name, seg_lens=seg_lens):
                    out = module.chunked_sgmv_shrink(x, weights, bi)
                    atol, rtol = TOLERANCES[x.dtype]
                    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


RELEASE_REQUIRED_TESTS = [
    "ChunkedSgmvShrinkTest.test_dtypes",
    "ChunkedSgmvShrinkTest.test_shapes",
    "ChunkedSgmvShrinkTest.test_identity_permutation",
    "ChunkedSgmvShrinkTest.test_tile_geometry_axes",
    "ChunkedSgmvShrinkTest.test_empty_batch",
    "ChunkedSgmvShrinkVariantsTest.test_pipeline_long_segment_dtypes",
    "ChunkedSgmvShrinkVariantsTest.test_variants_match_reference",
]


if __name__ == "__main__":
    unittest.main()

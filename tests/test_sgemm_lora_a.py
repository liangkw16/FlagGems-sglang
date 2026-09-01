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

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("sgemm_lora_a")


class BatchInfo:
    def __init__(
        self, seg_indptr, weight_indices, permutation=None, max_len=None
    ):
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices
        self.permutation = permutation
        self.bs = len(weight_indices)
        if max_len is not None:
            self.max_len = max_len


def reference(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]
    out = torch.zeros(S, R, dtype=x.dtype, device=x.device)
    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)
        x_seg = x[rows].float()
        w = weights[w_idx].float()
        val = x_seg @ w.t()
        out[rows] = val.to(x.dtype)
    return out


def make_case(S, K, R, num_lora, num_segs, dtype, seed, with_perm):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(S, K, generator=g).to(dtype=dtype, device="cuda")
    weights = (torch.randn(num_lora, R, K, generator=g) * 0.05).to(
        dtype=dtype, device="cuda"
    )
    # random non-decreasing segment boundaries covering all S rows
    if num_segs > 1 and S > 1:
        cuts = sorted(
            torch.randint(1, S, (num_segs - 1,), generator=g).tolist()
        )
    else:
        cuts = []
    boundaries = [0] + cuts + [S]
    indptr = torch.tensor(boundaries, dtype=torch.int64, device="cuda")
    widx = torch.randint(
        0, num_lora, (num_segs,), generator=g, dtype=torch.int64
    ).to("cuda")
    perm = None
    if with_perm:
        perm = torch.randperm(S, generator=g).to(device="cuda")
    max_len = max(
        (b - a for a, b in zip(boundaries, boundaries[1:])), default=0
    )
    return x, weights, BatchInfo(indptr, widx, perm, max_len)


class TestSgemmLoraA(unittest.TestCase):
    TOL = {
        torch.float32: dict(rtol=1e-4, atol=1e-4),
        torch.float16: dict(rtol=1e-2, atol=1e-2),
        torch.bfloat16: dict(rtol=1.5e-2, atol=1.5e-2),
    }

    def _check(
        self,
        x,
        weights,
        info,
        stack_num=1,
        relax_iluvatar_fp32_deep=False,
    ):
        ref = reference(x, weights, info, stack_num)
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.sgemm_lora_a(x, weights, info, stack_num)
                self.assertEqual(out.shape, ref.shape)
                self.assertEqual(out.dtype, x.dtype)
                tol = self.TOL[x.dtype]
                if relax_iluvatar_fp32_deep and name == "iluvatar":
                    tol = dict(rtol=2e-4, atol=2e-4)
                torch.testing.assert_close(out.float(), ref.float(), **tol)

    def test_matrix(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for S, K, R, num_lora, segs, perm in [
                (128, 512, 64, 4, 3, False),
                (1024, 2048, 128, 8, 16, False),
                (256, 1024, 192, 3, 7, True),
                (4096, 4096, 256, 16, 32, True),
                (17, 33, 16, 2, 2, False),
                (1, 64, 32, 1, 1, False),
                (512, 128, 8, 4, 5, True),
                (256, 512, 65, 4, 5, True),
                (128, 320, 80, 3, 4, False),
                (64, 96, 129, 2, 3, True),
            ]:
                with self.subTest(dtype=dtype, S=S, K=K, R=R, perm=perm):
                    x, w, info = make_case(
                        S, K, R, num_lora, segs, dtype, S + K, perm
                    )
                    self._check(
                        x,
                        w,
                        info,
                        relax_iluvatar_fp32_deep=(
                            dtype is torch.float32 and K >= 4096
                        ),
                    )

    def test_stack_num(self):
        x, w, info = make_case(256, 512, 128, 4, 5, torch.float16, 9, True)
        # stack_num is reflected in weights' first dim already; call parity
        self._check(x, w, info, stack_num=2)

    def test_empty_and_invariance(self):
        x, w, info = make_case(64, 128, 32, 2, 2, torch.float32, 3, False)
        del info.max_len
        xc, wc = x.clone(), w.clone()
        self._check(x, w, info)
        self.assertTrue(torch.equal(x, xc))
        self.assertTrue(torch.equal(w, wc))
        # zero rows segments still produce zeros elsewhere
        empty = BatchInfo(
            torch.tensor([0, 0, 5, 5], device="cuda"),
            torch.tensor([0, 1, 0], device="cuda"),
            max_len=5,
        )
        self._check(x[:8], w, empty)

    def test_non_contiguous_inputs_and_metadata(self):
        S, K, R = 9, 33, 65
        x = torch.randn(S, 2 * K, device="cuda")[:, ::2]
        weights = torch.randn(2, R, 2 * K, device="cuda")[:, :, ::2]
        permutation = [2, 0, 1, 5, 3, 4, 8, 6, 7]
        self.assertFalse(x.is_contiguous())
        self.assertFalse(weights.is_contiguous())

        for metadata_dtype in (torch.int64, torch.int32):

            def strided(values):
                storage = torch.zeros(
                    2 * len(values), dtype=metadata_dtype, device="cuda"
                )
                storage[::2] = torch.tensor(
                    values, dtype=metadata_dtype, device="cuda"
                )
                return storage[::2]

            info = BatchInfo(
                strided([0, 4, S]),
                strided([1, 0]),
                strided(permutation),
                max_len=5,
            )
            self.assertEqual(info.seg_indptr.stride(), (2,))
            self.assertEqual(info.weight_indices.stride(), (2,))
            self.assertEqual(info.permutation.stride(), (2,))
            self._check(x, weights, info)

    def test_zero_dimensions(self):
        cases = [
            (
                torch.randn(0, 4, device="cuda"),
                torch.randn(1, 3, 4, device="cuda"),
                BatchInfo(
                    torch.tensor([0, 0], device="cuda"),
                    torch.tensor([0], device="cuda"),
                    max_len=0,
                ),
            ),
            (
                torch.randn(2, 4, device="cuda"),
                torch.randn(1, 3, 4, device="cuda"),
                BatchInfo(
                    torch.tensor([0], device="cuda"),
                    torch.empty(0, dtype=torch.int64, device="cuda"),
                    max_len=0,
                ),
            ),
            (
                torch.randn(2, 0, device="cuda"),
                torch.randn(1, 3, 0, device="cuda"),
                BatchInfo(
                    torch.tensor([0, 2], device="cuda"),
                    torch.tensor([0], device="cuda"),
                    max_len=2,
                ),
            ),
            (
                torch.randn(2, 4, device="cuda"),
                torch.randn(1, 0, 4, device="cuda"),
                BatchInfo(
                    torch.tensor([0, 2], device="cuda"),
                    torch.tensor([0], device="cuda"),
                    max_len=2,
                ),
            ),
        ]
        for x, weights, info in cases:
            self._check(x, weights, info)


if __name__ == "__main__":
    unittest.main()
